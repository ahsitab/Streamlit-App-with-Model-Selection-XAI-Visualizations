import streamlit as st
import os
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import lime
from lime import lime_image
from captum.attr import LayerGradCam, LayerAttribution, Lime
from captum.attr import visualization as viz

# Set up the app
st.set_page_config(layout="wide")
st.title("XAI Visualizations for Image Classification")

# Sidebar - Model Selection
st.sidebar.header("Model Selection")
model_files = [f for f in os.listdir("models") if f.endswith(".pth")]
selected_model = st.sidebar.selectbox("Choose a model", model_files)

# Sidebar - Image Input
st.sidebar.header("Image Input")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
use_sample = st.sidebar.checkbox("Use sample image")

# Sample images
sample_images = {
    "Apple": "sample_images/apple.jpg",
    "Banana": "sample_images/banana.jpg",
    "Orange": "sample_images/orange.jpg"
}

# Load model function
def load_model(model_path):
    # Implement your model loading logic here
    # This will vary based on how your models were saved
    model = torch.load(model_path)
    model.eval()
    return model

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Generate predictions
def predict(image_tensor, model):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probs, 3)
    return top_probs.numpy()[0], top_classes.numpy()[0]

# XAI Methods
def generate_grad_cam(model, image_tensor, target_layer):
    grad_cam = LayerGradCam(model, target_layer)
    attributions = grad_cam.attribute(image_tensor, target=0)
    return attributions

def generate_lime_explanation(model, image, top_class):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance np.array(image), 
                                          lambda x: model(preprocess_image(Image.fromarray(x.astype('uint8'))).numpy(),
                                          top_labels=3, 
                                          hide_color=0, 
                                          num_samples=1000)
    temp, mask = explanation.get_image_and_mask(top_class, positive_only=True, num_features=5, hide_rest=False)
    return temp, mask

# Main app logic
def main():
    if not model_files:
        st.error("No models found in the 'models' directory. Please add your trained models.")
        return
    
    try:
        model = load_model(os.path.join("models", selected_model))
        st.sidebar.success(f"Loaded model: {selected_model}")
        
        # Display model metadata
        st.sidebar.subheader("Model Info")
        st.sidebar.text(f"Architecture: {model.__class__.__name__}")
        st.sidebar.text(f"Checkpoint: {selected_model}")
        
        # Image handling
        if use_sample:
            sample_choice = st.sidebar.selectbox("Choose sample image", list(sample_images.keys()))
            image_path = sample_images[sample_choice]
            image = Image.open(image_path)
        elif uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            st.info("Please upload an image or select a sample image")
            return
            
        # Display original image
        st.subheader("Input Image")
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Preprocess and predict
        image_tensor = preprocess_image(image)
        probs, classes = predict(image_tensor, model)
        
        # Display predictions
        st.subheader("Predictions")
        col1, col2, col3 = st.columns(3)
        class_names = ["Fresh", "Rotten", "Formalin-mixed"]  # Update with your actual class names
        
        for i, (prob, cls) in enumerate(zip(probs, classes)):
            with eval(f"col{i+1}"):
                st.metric(label=class_names[cls], value=f"{prob*100:.2f}%")
        
        predicted_class = classes[0]
        
        # XAI Visualizations
        st.subheader("Explainable AI Visualizations")
        
        # Grad-CAM
        target_layer = model.layer4[-1]  # Update with your model's target layer
        grad_cam_attr = generate_grad_cam(model, image_tensor, target_layer)
        grad_cam_vis = viz.visualize_image_attr(grad_cam_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                                               np.array(image.resize((224, 224))),
                                               method="blended_heat_map",
                                               sign="positive",
                                               show_colorbar=True)
        
        # LIME
        lime_temp, lime_mask = generate_lime_explanation(model, image, predicted_class)
        
        # Display visualizations in a grid
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(grad_cam_vis)
            st.caption("Grad-CAM Visualization")
        with col2:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(lime_temp)
            ax.imshow(lime_mask, alpha=0.5, cmap='viridis')
            st.pyplot(fig)
            st.caption("LIME Visualization")
            
        # Add more XAI methods (Grad-CAM++, Eigen-CAM, Ablation-CAM) similarly
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()