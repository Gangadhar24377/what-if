# data/data_loader.py
import torch
from torchvision import transforms, datasets
from PIL import Image

class ImageDataLoader:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_image(self, image_path):
        """Load and preprocess a single image."""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)

    def inverse_transform(self, tensor):
        """Convert tensor back to displayable image."""
        inverse_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        return inverse_normalize(tensor.squeeze()).clamp(0, 1)