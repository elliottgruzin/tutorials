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
        down_block_types=[
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D"
        ],
        up_block_types=[
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ],
        block_out_channels=(64, 128, 128, 128),
        cross_attention_dim=768
    )

def inference(noise, model, scheduler, encoded_text, guidance_scale = 10):
    model.eval()

    current_noise = torch.clone(noise)

    with torch.no_grad():
        for step in tqdm(scheduler.timesteps):
            noise_pred = model(current_noise, step, encoded_text).sample
            if guidance_scale != 0:
                unconditional_noise_pred = model(current_noise, step, torch.zeros(encoded_text.size()).to(accelerator.device)).sample
                noise_pred = torch.lerp(unconditional_noise_pred, noise_pred, guidance_scale)
            current_noise = scheduler.step(noise_pred, step, current_noise).prev_sample

    generated_images = current_noise
    return generated_images

def training_loop(model, dataloader, encoder, scheduler, criterion, optimizer):
    losses = []
    model.train()
    for batch in tqdm(dataloader):
        images = batch['images']
        token_ids = batch['text']
        encoded = encoder(token_ids).last_hidden_state
        
        # classifier-free guidance needs the model to learn how to predict without token info
        if np.random.random() < 0.1:
            encoded = torch.zeros(encoded.size()).to(accelerator.device)

        noise = torch.randn(images.shape).to(accelerator.device)
        bs = images.shape[0]

        timesteps = torch.randint(0, 1000 - 1, (bs,), dtype=torch.long).to(accelerator.device)
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
    scheduler = DDPMScheduler()

    dataset = load_and_transform()
    dataloader = get_dataloader(tokenizer, dataset, batch_size=32)
    
    model, optim, scheduler, encoder, dataloader = accelerator.prepare(model, optim, scheduler, encoder, dataloader)

    inference_noise = torch.randn(4, 3, 64, 64).to(accelerator.device)
    inference_tokenized_text = tokenizer(inference_text, padding='max_length', max_length=tokenizer.model_max_length, return_tensors='pt').to(accelerator.device)
    inference_encoded_text = encoder(**inference_tokenized_text).last_hidden_state

    for i in tqdm(range(epochs)):
        # epoch_losses = training_loop(model, dataloader, encoder, scheduler, criterion, optim)
        # mean_epoch_loss = np.mean(epoch_losses)
        model.eval()

        if i % 5 == 0:
            generated_images = inference(inference_noise, model, scheduler, inference_encoded_text)
            save_image(generated_images, f'epoch_{epoch}.png')
            save_losses(mean_epoch_loss, f'losses_{i}.png')

        print(f'Epoch {i} loss: {mean_epoch_loss.item()}')

    model.save_pretrained('trained_model')

if __name__=="__main__":
    main()