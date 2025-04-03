from PIL import Image

import torch
from torch.optim import AdamW
from torch.nn import MSELoss

import torchvision
from torchvision import transforms

from transformers import CLIPTextModel, CLIPTokenizerFast

from tqdm import tqdm

from diffusers import UNet2DConditionModel
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
import accelerate

import numpy as np

from datasets import load_dataset

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    print("Using CUDA")
    device = "cuda"

def create_image_grid(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    grid = torchvision.utils.make_grid(images, nrow=2)
    grid = grid.detach().cpu().permute(1, 2, 0).numpy() * 255
    grid_im = Image.fromarray(np.array(grid).astype(np.uint8))
    return grid_im

def save_image(images, path):
    grid = create_image_grid(images)
    grid.save(path)

image_preprocessor = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
]
)

def preprocess(ds):
    images = [image_preprocessor(example.convert('RGB')) for example in ds['image']]
    return {"images": images, 'text': ds['text']}

def conditional_inference(noise, model, scheduler, encoded_text):
    model.eval()

    current_noise = torch.clone(noise)

    with torch.no_grad():
        for step in tqdm(scheduler.timesteps):
            noise_pred = model(current_noise, step, encoded_text).sample
            current_noise = scheduler.step(noise_pred, step, current_noise).prev_sample

    generated_images = current_noise
    return generated_images


def save_losses(losses, path):
    plt.plot(losses)
    plt.savefig(path)

dataset = load_dataset("jiovine/pixel-art-nouns-2k")["train"]
dataset.set_transform(preprocess)

criterion = MSELoss()


encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")



text_to_image_denoiser = UNet2DConditionModel(
    sample_size=128,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    down_block_types=[
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D"
    ],
    up_block_types=[
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    ],
    block_out_channels=(64, 128, 256, 256),
    cross_attention_dim=768
)

scheduler = DDPMScheduler(beta_schedule='scaled_linear')

def collate_fn(examples):
    tokenized = tokenizer(
        [example['text'] for example in examples],
        padding='max_length', max_length=tokenizer.model_max_length, return_tensors='pt'
        ).input_ids
    images = torch.stack([example['images'] for example in examples])
    images = images.to(memory_format=torch.contiguous_format).float()
    return {"images": images, 'text': tokenized}

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


optimizer = AdamW(text_to_image_denoiser.parameters(), lr=1e-4)
accelerator = accelerate.Accelerator(mixed_precision='fp16')
text_to_image_denoiser, optimizer, train_dataloader, scheduler, tokenizer, encoder = accelerator.prepare(text_to_image_denoiser, optimizer, train_dataloader, scheduler, tokenizer, encoder)
num_train_timesteps = 1000

def conditional_generation_training_loop(model, optimizer, train_dataloader, scheduler, num_epochs, encoder):

    inference_text = [
        'a character with round green glasses, a bull-shaped head and a red-colored body on a cool background',
        'a character with no glasses, a robot-shaped head and a teal-colored body on a warm background',
        'a character with square pink and red glasses, a head with afro hair and a hotbrown-colored body on a warm background',
        'a character with round pink sunglasses, a toaster-shaped head and a redpinkish-colored body on a grey background',
    ]

    inference_tokenized_text = tokenizer(inference_text, padding='max_length', max_length=tokenizer.model_max_length, return_tensors='pt').to(device)
    inference_encoded_text = encoder(**inference_tokenized_text).last_hidden_state

    model.train()
    encoder.requires_grad_(False)

    eval_set = torch.randn(4, 3, 128, 128).to(device)

    for epoch in range(num_epochs):

        losses = []
        model.train()
        for batch in tqdm(train_dataloader):
            images = batch['images']
            token_ids = batch['text']
            encoded = encoder(token_ids).last_hidden_state
            noise = torch.randn(images.shape, device=device)
            bs = images.shape[0]

            timesteps = torch.randint(0, num_train_timesteps - 1, (bs,), device=device, dtype=torch.long)
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            predicted_noise = model(noisy_images, timesteps, encoded).sample

            loss = criterion(predicted_noise, noise)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        loss = np.mean(losses)
        model.eval()

        if epoch % 5 == 0:
            generated_images = conditional_inference(eval_set, model, scheduler, inference_encoded_text)
            save_image(generated_images, f'epoch_{epoch}.png')
            save_losses(losses, f'losses_{epoch}.png')

        print(f'Epoch {epoch} loss: {loss.item()}')

conditional_generation_training_loop(text_to_image_denoiser, optimizer, train_dataloader, scheduler, 35, encoder)
