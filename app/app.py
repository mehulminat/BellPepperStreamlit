import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image
from utils import set_background, classify

set_background('bgs/bg_2.png')

def main():
    
    st.title("Bell Pepper Leaf Disease Classification")
    st.header('Please upload a Bell Pepper Leaf Image')
    file = st.file_uploader("", type=['jpeg', 'jpg', 'png'])
    model = load_model('model/BellPepperMobileNet.h5')

    classes = ['Bell pepper Bacterial Spot', 'Bell pepper Healthy']

    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        predicted_class, confidence = classify(image, model, classes)

        st.write("## {}".format(predicted_class))
        st.write("### score: {}".format(int(confidence * 10)/ 10))


if __name__ == "__main__":
    main()