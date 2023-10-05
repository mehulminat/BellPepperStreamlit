import base64
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

def set_background(image_file):
    with open(image_file, 'rb') as f:
        image_data = f.read()
    b64_encoded = base64.b64encode(image_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image : url(data:image/png; base64, {b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, classes):

    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32)/ 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)

    predicted_class = classes[np.argmax(prediction[0])].lower()
    confidence = round(100*(np.max(prediction[0])), 3)

    return predicted_class, confidence