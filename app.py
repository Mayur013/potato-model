import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Load your model
MODEL = tf.keras.models.load_model("2.keras")

CLASS_NAMES = ['Potato Early blight', 'Potato Late blight', 'Potato healthy']

st.title("Potato Disease Classification")

st.write("Upload an image of a potato leaf to classify the disease.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

def read_file_as_image(data) -> np.array:
    image = np.array(Image.open(BytesIO(data)))
    return image

if uploaded_file is not None:
    image = read_file_as_image(uploaded_file.read())
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        confidence=confidence*100
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")
