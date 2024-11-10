import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class ImageDataLoader:
    def __init__(self):
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to expected input size for ResNet
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_image(self, image_file):
        """Load and preprocess a single image from a file-like object."""
        try:
            image = Image.open(image_file).convert('RGB')  # Ensure RGB
            print(f"Loaded image with size: {image.size}")  # Debug log
            return self.transform(image).unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"Error loading image: {type(e).__name__}: {str(e)}")
            raise ValueError(f"Failed to load and preprocess image: {type(e).__name__}: {str(e)}")
        
    def inverse_transform(self, tensor):
        """Convert tensor back to displayable image (undo normalization)."""
        try:
            inverse_normalize = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],  # Inverse normalization
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
            return inverse_normalize(tensor.squeeze()).clamp(0, 1)  # Remove batch dimension, clamp values
        except Exception as e:
            print(f"Error during inverse transformation: {type(e).__name__}: {str(e)}")
            raise ValueError(f"Failed to inverse transform image: {type(e).__name__}: {str(e)}")
