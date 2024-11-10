import json
from torchvision.models import ResNet50_Weights

def ensure_imagenet_classes():
    try:
        with open('imagenet_classes.json', 'r') as f:
            class_labels = json.load(f)
        
        if len(class_labels) == 1000:
            print("imagenet_classes.json has 1000 classes. No issues found.")
            return class_labels
        else:
            print(f"Warning: imagenet_classes.json has {len(class_labels)} classes, expected 1000.")
    except (FileNotFoundError, json.JSONDecodeError):
        print("imagenet_classes.json not found or invalid. Regenerating file...")

    # Regenerate imagenet_classes.json
    class_labels = ResNet50_Weights.DEFAULT.meta['categories']
    with open('imagenet_classes.json', 'w') as f:
        json.dump(class_labels, f)
    print("imagenet_classes.json has been regenerated with 1000 classes.")

    return class_labels

# Ensure class labels are valid
class_labels = ensure_imagenet_classes()
