import streamlit as st
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# Import model inference / visualization modules
from src.visualize import overlay

st.title("Leaf Segmentation App")
st.write("Upload a leaf image to generate segmentation masks.")

# Download checkpoint at startup
@st.cache_resource
def load_checkpoint():
    return hf_hub_download("ForgeCrewz/leaf-maskrcnn", "maskrcnn_leaf.pth")

checkpoint_path = load_checkpoint()

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing segmentations...")
