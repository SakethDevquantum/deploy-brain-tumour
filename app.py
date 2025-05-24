import streamlit as st
import requests
from PIL import Image


API_URL = "https://your-fastapi-app.onrender.com/predict/"

st.title("Brain Tumor Classification using ViT")
st.write("Upload a brain MRI image (grayscale) to predict the tumor class.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        with st.spinner("Sending to model..."):
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Predicted Tumor Type: **{prediction}**")
        else:
            st.error("Prediction failed: " + response.text)
