import accelerate
import torch
import numpy as np
from tqdm import tqdm

from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizerFast

from data.celebrity import load_and_transform, get_dataloader, inference_text
from utils import save_image, save_losses, create_image_grid

accelerator = accelerate.Accelerator()

def load_text_models():
    encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
    return encoder, tokenizer

def generate_model():
    return UNet2DConditionModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256),
        cross_attention_dim=768
    )

def inference(noise, model, scheduler, encoded_text, guidance_scale=5):
    model.eval()
    current_noise = torch.clone(noise)

    with torch.no_grad():
        for step in tqdm(scheduler.timesteps):
            noise_pred = model(current_noise, step, encoded_text).sample
            if guidance_scale != 0:
                unconditional_noise_pred = model(
                    current_noise, step, torch.zeros_like(encoded_text)
                ).sample
                noise_pred = torch.lerp(unconditional_noise_pred, noise_pred, guidance_scale)
            current_noise = scheduler.step(noise_pred, step, current_noise).prev_sample

    return current_noise

def training_loop(model, dataloader, encoder, scheduler, criterion, optimizer):
    losses = []
    model.train()
    
    for batch in tqdm(dataloader):
        with accelerator.accumulate(model):
            images = batch['images']
            token_ids = batch['text']
            encoded = encoder(token_ids).last_hidden_state

            # Synchronized classifier-free guidance dropout
            if accelerator.is_main_process:
                do_zero = torch.rand(1).item() < 0.1
                do_zero_tensor = torch.tensor(do_zero, dtype=torch.bool, device=accelerator.device)
            else:
                do_zero_tensor = torch.tensor(False, dtype=torch.bool, device=accelerator.device)

            if torch.distributed.is_initialized():
                torch.distributed.broadcast(do_zero_tensor, src=0)

            if do_zero_tensor.item():
                encoded = torch.zeros_like(encoded)


            noise = torch.randn_like(images)
            bs = images.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps - 1, (bs,), dtype=torch.long, device=images.device)
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            predicted_noise = model(noisy_images, timesteps, encoded).sample

            loss = criterion(predicted_noise, noise)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
    return losses

def main():
    epochs = 1
    encoder, tokenizer = load_text_models()
    model = generate_model()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    dataset = load_and_transform()
    dataloader = get_dataloader(tokenizer, dataset, batch_size=32)

    model, optim, scheduler, encoder, dataloader = accelerator.prepare(
        model, optim, scheduler, encoder, dataloader
    )

    inference_noise = torch.randn(4, 3, 64, 64).to(accelerator.device)
    inference_tokenized_text = tokenizer(
        inference_text, padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt'
    ).to(accelerator.device)
    inference_encoded_text = encoder(**inference_tokenized_text).last_hidden_state

    for i in tqdm(range(epochs)):
        accelerator.print(f"Rank {accelerator.process_index}: Starting epoch {i}")
        epoch_losses = training_loop(model, dataloader, encoder, scheduler, criterion, optim)
        mean_epoch_loss = np.mean(epoch_losses)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if i % 5 == 0:
                generated_images = inference(inference_noise, model, scheduler, inference_encoded_text)
                save_image(generated_images, f'epoch_{i}.png')
                save_losses(mean_epoch_loss, f'losses_{i}.png')
            print(f'Epoch {i} loss: {mean_epoch_loss:.4f}')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained('trained_model')

if __name__ == "__main__":
    main()
