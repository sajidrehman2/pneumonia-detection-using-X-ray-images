import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Define your custom Swish activation function if it's not already loaded
def swish_activation(x):
    return x * tf.nn.sigmoid(x)

# Load your trained model
model = load_model('final_model.keras', custom_objects={'swish_activation': swish_activation})

# Set up the Streamlit app with custom styles
st.set_page_config(page_title="Pneumonia Detection App", page_icon="🩺", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Page background */
        .stApp {
            background-color: #f4f7f9;
        }
        /* Title styling */
        .title {
            color: #1a73e8;
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
        }
        /* Subheading styling */
        .subheading {
            color: #444444;
            font-size: 1.2em;
            text-align: center;
            margin-bottom: 1em;
        }
        /* File uploader styling */
        .stFileUploader label {
            font-size: 1.1em;
            color: #1a73e8;
            font-weight: bold;
        }
        /* Button styling */
        div.stButton > button {
            background-color: #1a73e8;
            color: white;
            font-size: 1.1em;
            padding: 0.5em 1em;
            border-radius: 8px;
            border: none;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #0f59c7;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and description
st.markdown('<h1 class="title">Pneumonia Detection Model</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheading">This app uses a CNN to classify chest X-rays as either "Normal" or "Pneumonia".</p>', unsafe_allow_html=True)

# Upload image functionality
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    img = Image.open(uploaded_file).convert('RGB')  # Convert to RGB to ensure 3 color channels
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Add a styled prediction button
    if st.button("Make Prediction"):
        # Prepare the image for prediction
        img = img.resize((150, 150))  # Resize the image to match the model input size
        img_array = np.array(img) / 255.0  # Normalize the image

        # Reshape the image array to match the model's expected input shape
        if model.input_shape == (None, 3, 150, 150):  # For 'channels_first' format
            img_array = np.transpose(img_array, (2, 0, 1))  # Rearrange to (channels, height, width)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)

        # Probability of the two classes (Pneumonia and Normal)
        pneumonia_prob = prediction[0][0]
        normal_prob = 1 - pneumonia_prob

        # Display the prediction results
        st.markdown(
            f'<p style="color: red; font-size: 1.5em; font-weight: bold; text-align: center;">Pneumonia: {pneumonia_prob * 100:.2f}%</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="color: green; font-size: 1.5em; font-weight: bold; text-align: center;">Normal: {normal_prob * 100:.2f}%</p>',
            unsafe_allow_html=True,
        )

        # Display overall prediction
        if pneumonia_prob > 0.5:
            st.markdown(
                '<p style="color: red; font-size: 1.8em; font-weight: bold; text-align: center;">Prediction: Pneumonia</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p style="color: green; font-size: 1.8em; font-weight: bold; text-align: center;">Prediction: Normal</p>',
                unsafe_allow_html=True,
            )
