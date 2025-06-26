import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# ---------------------- PAGE CONFIG ---------------------- #
st.set_page_config(
    page_title="Skin Tumor Classifier",
    layout="wide",
    page_icon="🧬"
)

# ---------------------- CUSTOM STYLES ---------------------- #
st.markdown("""
    <style>
        .main-title {
            font-size: 48px;
            color: #FF6F61;
            text-align: center;
            font-weight: bold;
        }
        .section-title {
            color: #4CAF50;
            font-size: 28px;
            margin-top: 2rem;
        }
        .content-text {
            font-size: 18px;
            line-height: 1.6;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- HELPER FUNCTIONS ---------------------- #
@st.cache_resource
def load_model():
    """Load and return the trained TensorFlow model."""
    return tf.keras.models.load_model("my_model.keras")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize the image for model prediction."""
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# ---------------------- PAGE 1: INTRODUCTION ---------------------- #
def show_introduction():
    st.markdown('<h1 class="main-title">🧬 Skin Tumor Classification Project</h1>', unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">🎯 Project Objectives</h2>', unsafe_allow_html=True)
    st.markdown('<p class="content-text">This project uses deep learning to classify skin tumor images into 10 distinct categories based on their visual characteristics.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/model prediction.png 2.png", caption="Model Prediction Sample")
    with col2:
        st.image("images/model prediction.png", caption="Model Prediction Sample")

    st.markdown('<h2 class="section-title">📦 Dataset Information</h2>', unsafe_allow_html=True)
    st.markdown('<p class="content-text">The dataset includes preprocessed and augmented images of skin tumors, covering categories like Melanoma, Basal Cell Carcinoma, and more.</p>', unsafe_allow_html=True)
    st.image("images/data images.png", caption="Sample Data Across Categories")

    st.markdown('<h2 class="section-title">🧰 Tech Stack</h2>', unsafe_allow_html=True)
    st.markdown('<p class="content-text">TensorFlow, NumPy, Pandas, PIL, Matplotlib, Streamlit</p>', unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">📈 Model Training & Evaluation</h2>', unsafe_allow_html=True)
    st.markdown('<p class="content-text">The model was trained using TensorFlow. Below are the training vs. validation accuracy and loss curves.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/Model Accuracy over epochs.png", caption="Accuracy Over Epochs")
    with col2:
        st.image("images/Model loss over epochs.png", caption="Loss Over Epochs")

    st.markdown('<h2 class="section-title">🧮 Confusion Matrix</h2>', unsafe_allow_html=True)
    st.markdown('<p class="content-text">The confusion matrix highlights model performance per class, revealing accuracy and misclassifications.</p>', unsafe_allow_html=True)
    st.image("images/confusion matrix.png", caption="Confusion Matrix")

# ---------------------- PAGE 2: DEMO ---------------------- #
def show_demo():
    st.markdown('<h1 class="main-title">🧪 Skin Tumor Prediction Demo</h1>', unsafe_allow_html=True)
    st.markdown('<p class="content-text">Upload a skin image below to predict the tumor type. The model outputs the most likely class along with a confidence score.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload an image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.info("⏳ Processing your image. Please wait...")
        image = Image.open(uploaded_file).convert('RGB')
        model = load_model()
        processed_img = preprocess_image(image)

        prediction = model.predict(processed_img)
        class_names = [
            'Actinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma', 'Melanoma',
            'Nevus', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis',
            'Squamous Cell Carcinoma', 'Vascular Lesion', 'Tinea Ringworm Candidiasis'
        ]
        pred_index = np.argmax(prediction)
        pred_class = class_names[pred_index]
        confidence = 100 * prediction[0][pred_index]

        st.balloons()

        # Layout results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🖼️ Uploaded Image")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### 🔍 Prediction Result")
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(f"{pred_class}\nConfidence: {confidence:.2f}%", fontsize=12, color='green')
            st.pyplot(fig)

        st.success(f"✅ Prediction: **{pred_class}** with **{confidence:.2f}%** confidence")

# ---------------------- MAIN APP ---------------------- #
def main():
    st.sidebar.title("🔎 Navigation")
    menu = {
        "📘 Introduction": show_introduction,
        "🚀 Try the Demo": show_demo
    }

    choice = st.sidebar.radio("Go to", list(menu.keys()))
    menu[choice]()

if __name__ == "__main__":
    main()
