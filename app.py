import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("vehicle_classifier_mobilenet.keras")

st.title("Vehicle Detection (Car vs Bike)")

uploaded_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0][0]
    label = "Car" if pred>0.5 else "Bike"
    
    st.success(f"Prediction: {label}")
