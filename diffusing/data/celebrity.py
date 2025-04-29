import torch
from torchvision import transforms
from datasets import load_dataset

image_preprocessor = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
]
)

def preprocess(ds):
    images = [image_preprocessor(example.convert('RGB')) for example in ds['image']]
    return {"images": images, 'text': ds["text"]}

def load_and_transform():
    dataset = load_dataset("irodkin/celeba_with_llava_shorter_captions")["train"]
    dataset.set_transform(preprocess)
    return dataset

def get_dataloader(tokenizer, dataset, batch_size):
    def collate_fn(examples):
        tokenized = tokenizer(
            [example['text'] for example in examples],
            padding='max_length', max_length=tokenizer.model_max_length, return_tensors='pt', truncation=True
            ).input_ids
        images = torch.stack([example['images'] for example in examples])
        images = images.to(memory_format=torch.contiguous_format).float()
        return {"images": images, 'text': tokenized}

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

inference_text = [
        'The image features a beautiful woman with a large nose, wide eyes, and blonde hair. She is wearing a scarf around her neck and has a smile on her face.',
        'The image features a young woman with long hair, wearing a black hat and sunglasses. Her eyes are wide open, and her eyebrows are thick.',
        'The image features a man with a large nose, wide eyes, thick lips, and a wide mouth. He is wearing glasses and is described as a "very cute" man. He is standing in front of a building, and his overall appearance is described as wide.',
        'The image features a bald man with glasses, a round face, a large nose, and small eyes. He is wearing glasses and has a bald head, which suggests that he might have a receding hairline or no hair at all.',
    ]