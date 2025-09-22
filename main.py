import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


model = tf.keras.models.load_model("fish_classifier.h5")

img_height, img_width = 128, 128
class_names = ['Black Sea Sprat', 'Gilt-Head Bream', 'Horse Mackerel',
               'Red Mullet', 'Red Sea Bream', 'Sea Bass',
               'Shrimp', 'Striped Red Mullet', 'Trout']

# STREAMLIT INTERFACE (with image drop box too!)
st.title("Fish Market Classifier")
st.write("Upload an image of a fish, and the model will predict the species!")
st.write("It should be one of these species: 'Black Sea Sprat', 'Gilt-Head Bream', 'Horse Mackerel','Red Mullet', 'Red Sea Bream', 'Sea Bass','Shrimp', 'Striped Red Mullet', 'Trout'")
st.write("The images have to be fishes laid on their side, or the model gets confused!")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction made here with a confidence score
    prediction = model.predict(img_array)
    score = np.max(prediction)
    class_idx = np.argmax(prediction)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write(f"### I'm confident **{score*100:.2f}%** that this is a: **{class_names[class_idx]}**")
    st.write("How did I do :o?")