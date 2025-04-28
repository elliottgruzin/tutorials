import accelerate
import torch
import numpy as np
from tqdm import tqdm

from diffusers import UNet2DModel, DDPMScheduler

from data.celebrity import load_and_transform, get_dataloader, inference_text
from utils import save_image, save_losses, create_image_grid

accelerator = accelerate.Accelerator()

def generate_model():
    return UNet2DModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 64, 128, 128),
        # block_out_channels=(128, 128, 256, 256),
    )

def inference(noise, model, scheduler):
    model.eval()
    current_noise = torch.clone(noise)

    with torch.no_grad():
        for step in tqdm(scheduler.timesteps):
            noise_pred = model(current_noise, step).sample
            current_noise = scheduler.step(noise_pred, step, current_noise).prev_sample
    return current_noise

def training_loop(model, dataloader, scheduler, criterion, optimizer):
    losses = []
    model.train()
    
    for batch in tqdm(dataloader):
        with accelerator.accumulate(model):
            images = batch['images']
            noise = torch.randn_like(images)
            bs = images.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps - 1, (bs,), dtype=torch.long, device=images.device)
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            predicted_noise = model(noisy_images, timesteps).sample

            loss = criterion(predicted_noise, noise)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
    return losses

def main():
    epochs = 1
    model = generate_model()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    dataset = load_and_transform()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )

    model, optim, scheduler, dataloader = accelerator.prepare(
        model, optim, scheduler, dataloader
    )

    inference_noise = torch.randn(4, 3, 64, 64).to(accelerator.device)
    for i in tqdm(range(epochs)):
        accelerator.print(f"Rank {accelerator.process_index}: Starting epoch {i}")
        epoch_losses = training_loop(model, dataloader, scheduler, criterion, optim)
        mean_epoch_loss = np.mean(epoch_losses)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if i % 5 == 0:
                generated_images = inference(inference_noise, model, scheduler)
                save_image(generated_images, f'epoch_{i}.png')
                save_losses(mean_epoch_loss, f'losses_{i}.png')
            print(f'Epoch {i} loss: {mean_epoch_loss:.4f}')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained('trained_model')

if __name__ == "__main__":
    main()
