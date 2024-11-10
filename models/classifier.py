import torch
import torch.nn as nn
import torchvision.models as models

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def get_features(self, x, layer_name='layer4'):
        """Extract intermediate features from specified layer."""
        features = {}
        def hook_fn(module, input, output):
            features['output'] = output
            
        handle = getattr(self.model, layer_name).register_forward_hook(hook_fn)
        self.model(x)
        handle.remove()
        
        return features['output']
