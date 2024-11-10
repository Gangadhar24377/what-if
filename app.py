# app.py
import streamlit as st
import torch
import json
from models.classifier import ImageClassifier
from models.counterfactual import CounterfactualGenerator
from data.data_loader import ImageDataLoader
import matplotlib.pyplot as plt

def load_imagenet_labels():
    try:
        with open('imagenet_classes.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        from torchvision.models import ResNet50_Weights
        class_idx = ResNet50_Weights.DEFAULT.meta['categories']
        with open('imagenet_classes.json', 'w') as f:
            json.dump(class_idx, f)
        return class_idx

def get_prediction(model, image_tensor, class_labels):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_index = predicted.item()
    print(f"Predicted index: {predicted_index}")  # Debug statement to check predicted index range

    if predicted_index >= len(class_labels):
        raise KeyError(f"Predicted index {predicted_index} is out of bounds for class labels.")

    predicted_class = class_labels[predicted_index] # Access the dictionary using the predicted index
    return predicted_class, confidence.item()

def main():
    st.title("What-If: Image Counterfactual Generator")
    st.write("This app demonstrates explainable AI with image classification.")

    # Load class labels first
    st.write("Loading ImageNet labels...")
    class_labels = load_imagenet_labels()
    num_classes = len(class_labels)
    st.write(f"Loaded {num_classes} class labels.")

    @st.cache_resource
    def load_models():
        classifier = ImageClassifier(num_classes=num_classes)
        counterfactual_gen = CounterfactualGenerator(classifier)
        data_loader = ImageDataLoader()
        return classifier, counterfactual_gen, data_loader

    classifier, counterfactual_gen, data_loader = load_models()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            st.write("Loading image using ImageDataLoader...")
            image_tensor = data_loader.load_image(uploaded_file)
            st.write(f"Image loaded. Tensor shape: {image_tensor.shape}")

            # Get original prediction
            st.write("Getting original prediction...")
            original_class, confidence = get_prediction(classifier, image_tensor, class_labels)
            original_class = original_class   # Access the class label using the predicted index
            st.write(f"Original Classification: {original_class} (Confidence: {confidence:.2%})")

            # Target class selection
            target_class_index = st.selectbox(
                "Select target class for counterfactual",
                options=list(range(num_classes)),
                format_func=lambda x: class_labels[x]
            )
            target_class = class_labels[target_class_index]

            if st.button("Generate Counterfactual"):
                with st.spinner("Generating counterfactual explanation..."):
                    try:
                        st.write("Generating counterfactual image...")
                        # Generate counterfactual
                        counterfactual = counterfactual_gen.generate(image_tensor, target_class, class_labels)
                        st.write("Counterfactual generation complete.")

                        # Calculate difference
                        st.write("Calculating difference between original and counterfactual...")
                        difference = torch.abs(counterfactual - image_tensor)

                        # Display results
                        col1, col2, col3 = st.columns(3)

                        # Original image
                        original_img = data_loader.inverse_transform(image_tensor)
                        col1.image(
                            original_img.permute(1, 2, 0).numpy(),
                            caption="Original Image",
                            use_container_width=True
                        )

                        # Counterfactual image
                        counterfactual_img = data_loader.inverse_transform(counterfactual)
                        col2.image(
                            counterfactual_img.permute(1, 2, 0).numpy(),
                            caption="Counterfactual Image",
                            use_container_width=True
                        )

                        # Difference visualization
                        diff_img = data_loader.inverse_transform(difference * 3)  # Amplify differences
                        col3.image(
                            diff_img.permute(1, 2, 0).numpy(),
                            caption="Changes Made (Amplified)",
                            use_container_width=True
                        )

                        # Add analysis
                        st.markdown("### Analysis")
                        new_class, new_confidence = get_prediction(classifier, counterfactual, class_labels)
                        st.write(f"New Classification: {new_class} (Confidence: {new_confidence:.2%})")

                        # Calculate and display change metrics
                        pixel_change = torch.abs(counterfactual - image_tensor).mean().item()
                        st.write(f"Average pixel change: {pixel_change:.3f}")

                        # Show most significant changes
                        st.markdown("### Areas of Significant Change")
                        fig, ax = plt.subplots()
                        heatmap = torch.abs(counterfactual - image_tensor).mean(dim=1).squeeze()
                        im = ax.imshow(heatmap.numpy(), cmap='hot')
                        plt.colorbar(im)
                        ax.set_title("Change Heatmap")
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error during counterfactual generation: {str(e)}")

        except KeyError as e:
            st.error(f"Key error processing image: {str(e)}")
        except Exception as e:
            st.error(f"General error processing image: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    main()
