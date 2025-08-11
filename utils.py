import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def preprocess_image(image, size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return tensor * std + mean

def tensor_to_image(tensor):
    tensor = denormalize(tensor)
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (image * 255).astype(np.uint8)