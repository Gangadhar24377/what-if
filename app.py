# app.py
import streamlit as st
import torch
import torchvision.models as models
from PIL import Image
import numpy as np
from torchvision import transforms
import json
import matplotlib.pyplot as plt
from models.classifier import ImageClassifier
from models.counterfactual import CounterfactualGenerator
from data.data_loader import ImageDataLoader

# Load ImageNet class labels
with open('Projects/what-if/imagenet_classes.json', 'r') as f:
    class_labels = json.load(f)

def load_models():
    """Initialize models with pretrained weights."""
    classifier = ImageClassifier()
    counterfactual_gen = CounterfactualGenerator(classifier)
    return classifier, counterfactual_gen

def get_prediction(model, image_tensor):
    """Get model prediction and confidence."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

def generate_explanation(counterfactual_img, original_img, target_class, original_class):
    """Generate human-readable explanation of changes."""
    # Calculate differences in key areas (simplified example)
    diff = torch.abs(counterfactual_img - original_img).mean(dim=1)
    
    # Find areas of significant change
    significant_changes = torch.where(diff > diff.mean() + diff.std())
    
    # Generate explanation based on changes
    explanation = f"To change the classification from '{class_labels[original_class]}' to '{class_labels[target_class]}', the model made these changes:\n\n"
    
    # Analyze color changes
    color_diff = (counterfactual_img - original_img).mean(dim=(2, 3))
    for i, diff in enumerate(['red', 'green', 'blue']):
        if abs(color_diff[0][i]) > 0.1:
            direction = 'increased' if color_diff[0][i] > 0 else 'decreased'
            explanation += f"- {direction} {diff} intensity\n"
    
    # Analyze spatial changes
    spatial_diff = torch.abs(counterfactual_img - original_img).mean(dim=1)
    if spatial_diff[:, :spatial_diff.shape[1]//2].mean() > spatial_diff[:, spatial_diff.shape[1]//2:].mean():
        explanation += "- Modified features in the upper part of the image\n"
    else:
        explanation += "- Modified features in the lower part of the image\n"
    
    return explanation

def main():
    st.title("Explainable AI: Image Counterfactual Generator")
    st.write("""
    This app demonstrates explainable AI by showing what changes would be needed to make 
    the model classify an image differently. Upload an image and select a target class to see 
    the minimal changes required.
    """)

    # Initialize models
    classifier, counterfactual_gen = load_models()
    data_loader = ImageDataLoader()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Original Image", use_column_width=True)

        # Process image
        image_tensor = data_loader.load_image(uploaded_file)
        
        # Get original prediction
        original_class, confidence = get_prediction(classifier, image_tensor)
        st.write(f"Original Classification: {class_labels[original_class]} (Confidence: {confidence:.2%})")

        # Target class selection
        target_class = st.selectbox(
            "Select target class for counterfactual",
            options=list(range(len(class_labels))),
            format_func=lambda x: class_labels[x]
        )

        if st.button("Generate Counterfactual"):
            with st.spinner("Generating counterfactual explanation..."):
                # Generate counterfactual
                counterfactual = counterfactual_gen.generate(image_tensor, target_class)
                
                # Calculate difference
                difference = torch.abs(counterfactual - image_tensor)

                # Display results in columns
                col1, col2, col3 = st.columns(3)
                
                # Original image
                original_img = data_loader.inverse_transform(image_tensor)
                col1.image(
                    original_img.permute(1, 2, 0).numpy(),
                    caption="Original Image",
                    use_column_width=True
                )

                # Counterfactual image
                counterfactual_img = data_loader.inverse_transform(counterfactual)
                col2.image(
                    counterfactual_img.permute(1, 2, 0).numpy(),
                    caption="Counterfactual Image",
                    use_column_width=True
                )

                # Difference visualization
                diff_img = data_loader.inverse_transform(difference * 3)  # Amplify differences
                col3.image(
                    diff_img.permute(1, 2, 0).numpy(),
                    caption="Changes Made (Amplified)",
                    use_column_width=True
                )

                # Generate and display explanation
                explanation = generate_explanation(
                    counterfactual, 
                    image_tensor, 
                    target_class, 
                    original_class
                )
                st.markdown("### Explanation of Changes")
                st.write(explanation)

                # Feature importance visualization
                st.markdown("### Feature Importance Heatmap")
                fig, ax = plt.subplots()
                heatmap = torch.abs(counterfactual - image_tensor).mean(dim=1).squeeze()
                im = ax.imshow(heatmap, cmap='hot')
                plt.colorbar(im)
                ax.set_title("Areas of Significant Change")
                st.pyplot(fig)

                # Add detailed analysis
                st.markdown("### Statistical Analysis")
                pixel_change = torch.abs(counterfactual - image_tensor).mean().item()
                st.write(f"Average pixel change: {pixel_change:.3f}")
                
                # Confidence comparison
                new_class, new_confidence = get_prediction(classifier, counterfactual)
                st.write(f"New Classification: {class_labels[new_class]} (Confidence: {new_confidence:.2%})")

if __name__ == "__main__":
    main()