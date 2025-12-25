import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Indian Food Classification",
    layout="wide"
)

IMG_SIZE = 200

# --------------------------------------------------
# Load Class Names
# --------------------------------------------------
@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)

class_names = load_class_names()

# --------------------------------------------------
# Load SavedModel (NO ERRORS)
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "indian_food_resnet50_savedmodel"
    )
    return model

model = load_model()

# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------
def predict(image):
    img = preprocess_image(image)
    preds = model.predict(img, verbose=0)
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))
    return class_names[class_id], confidence

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🍛 Indian Food Image Classification")
st.markdown(
    "Upload **one or multiple images** to classify Indian food dishes using **ResNet50**."
)

uploaded_files = st.file_uploader(
    "Upload Food Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("🔍 Prediction Results")
    cols = st.columns(3)

    for idx, file in enumerate(uploaded_files):
        image = Image.open(file)
        label, confidence = predict(image)

        with cols[idx % 3]:
            st.image(image)
            st.markdown(
                f"""
                <div style="
                    padding:14px;
                    border-radius:12px;
                    background-color:#f4f6f8;
                    text-align:center;
                    box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                    <h4 style="margin-bottom:6px;">{label}</h4>
                    <p style="font-size:14px;">
                        Confidence: <b>{confidence:.2f}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("📌 Upload one or more food images to get predictions")
