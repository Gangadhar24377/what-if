import torch
from data.data_loader import ImageDataLoader
from models.classifier import ImageClassifier
from models.counterfactual import CounterfactualGenerator
from utils.visualization import plot_counterfactual
import torchvision.models as models
import json

def load_class_indices():
    """Load ImageNet class indices for easy reference"""
    try:
        with open('Projects/what-if/imagenet_classes.json', 'r') as f:
            return json.load(f)
    except:
        # If file doesn't exist, create it
        from torchvision.models import ResNet50_Weights
        class_idx = ResNet50_Weights.DEFAULT.meta['categories']
        with open('imagenet_classes.json', 'w') as f:
            json.dump(class_idx, f)
        return class_idx

def main():
    # Initialize components
    data_loader = ImageDataLoader()
    classifier = ImageClassifier()
    counterfactual_gen = CounterfactualGenerator(classifier)
    class_labels = load_class_indices()
    
    # Example test cases
    test_cases = [
        {
            "image_path": "path/to/dog.jpg",
            "target_class": class_labels.index("wolf"),  # Convert wolf to index
            "description": "Dog to Wolf transformation"
        },
        {
            "image_path": "path/to/housecat.jpg",
            "target_class": class_labels.index("tiger"),
            "description": "House cat to Tiger transformation"
        }
    ]
    
    # Process each test case
    for test in test_cases:
        print(f"\nProcessing: {test['description']}")
        
        # Load and process image
        try:
            image = data_loader.load_image(test['image_path'])
        except Exception as e:
            print(f"Error loading image: {e}")
            continue
            
        # Get original prediction
        with torch.no_grad():
            outputs = classifier(image)
            _, predicted = outputs.max(1)
            original_class = predicted.item()
            
        print(f"Original class: {class_labels[original_class]}")
        print(f"Target class: {class_labels[test['target_class']]}")
        
        # Generate counterfactual
        counterfactual = counterfactual_gen.generate(
            image, 
            test['target_class'],
            lambda_reg=0.1  # Adjust this to control magnitude of changes
        )
        
        # Calculate difference
        difference = torch.abs(counterfactual - image)
        
        # Visualize results
        original_img = data_loader.inverse_transform(image)
        counterfactual_img = data_loader.inverse_transform(counterfactual)
        diff_img = data_loader.inverse_transform(difference)
        
        # Plot and save results
        fig = plot_counterfactual(original_img, counterfactual_img, diff_img)
        fig.savefig(f"counterfactual_{test['description'].replace(' ', '_')}.png")
        print(f"Saved visualization to counterfactual_{test['description'].replace(' ', '_')}.png")
        
        # Get counterfactual prediction
        with torch.no_grad():
            outputs = classifier(counterfactual)
            _, predicted = outputs.max(1)
            cf_class = predicted.item()
            
        print(f"Counterfactual class: {class_labels[cf_class]}")
        print("-" * 50)

if __name__ == "__main__":
    main()