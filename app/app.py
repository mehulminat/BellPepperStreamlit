import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image
from utils import set_background, classify

set_background('./bgs/bg_2.png')

# def main():
    
#     st.title("Bell Pepper Leaf Disease Classification")
#     st.header('Please upload a Bell Pepper Leaf Image')
#     file = st.file_uploader("", type=['jpeg', 'jpg', 'png'])
#     model = load_model('model/BellPepperMobileNet.h5')

#     classes = ['Bell pepper Bacterial Spot', 'Bell pepper Healthy']

#     if file is not None:
#         image = Image.open(file).convert('RGB')
#         st.image(image, use_column_width=True)

#         predicted_class, confidence = classify(image, model, classes)

#         st.write("## {}".format(predicted_class))
#         st.write("### score: {}".format(int(confidence * 10)/ 10))


# if __name__ == "__main__":
#     main()

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "bgs/bg_2.png"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
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

