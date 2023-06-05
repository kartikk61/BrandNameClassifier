import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the model from h5 file
model = keras.models.load_model('model_resnet50.h5')

# Define class labels
class_labels = ['Audi','Lamborghini','Mercedes']

def preprocess_image(image):
    # Resize the image to match the input size of the model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the image by scaling pixel values to [0, 1]
    image = image / 255.0
    # Expand the dimensions of the image to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    return image

def classify_image(image):
    # Preprocess the image
    image = preprocess_image(image)
    # Make predictions using the model  
    predictions = model.predict(image)
    # Get the index of the predicted class
    predicted_class_index = np.argmax(predictions)
    # Get the predicted class label
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

# Streamlit app
st.title("Brand name classification using Image ")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Classify the image
    predicted_class = classify_image(image)

    # Display the predicted class
    st.write("Predicted Class:", predicted_class)
