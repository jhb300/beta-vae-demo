# Script implementation of the demo for integration with Streamlit

import streamlit as st
import cv2
from visualization.visualizer import Visualizer
import numpy as np

# Load the model
path_mnist = "./trained_models/mnist/"
viz_mnist = Visualizer(path_mnist)


# Define function to update image
def update_image(number_type, width, style1, style2, tilt, thickness):
    continuous = [width, style1, tilt, 0.0, 0.0, 0.0, thickness, 0.0, style2, 0.0]
    discrete = np.zeros(10, int)
    number_mapping = [2, 8, 9, 1, 6, 3, 5, 4, 7, 0]
    discrete[number_mapping.index(number_type)] = (
        3 # 3 for a good sharpness, fixed instead of ppf
    )
    img = viz_mnist.visualize(
        discrete_latent_parameters=discrete, continous_latent_parameters=continuous
    )
    return img


# Streamlit UI
st.title("Disentangled Variational Autoencoder Demo")

# Define sliders for each parameter
number_type = st.sidebar.slider("Number type", 0, 9, 4)
width = st.sidebar.slider("Width", -3.0, 3.0, 0.0, 0.1)
style1 = st.sidebar.slider("Style 1", -3.0, 3.0, 0.0, 0.1)
style2 = st.sidebar.slider("Style 2", -3.0, 3.0, 0.0, 0.1)
tilt = st.sidebar.slider("Tilt", -3.0, 3.0, 0.0, 0.1)
thickness = st.sidebar.slider("Thickness", -3.0, 3.0, 0.0, 0.1)

# Display the generated image
img = update_image(number_type, width, style1, style2, tilt, thickness)
# Using the Inter_Nearest interpolation for enhanced sharpness due to very low resolution.
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
st.image(img, caption="Generated Image", use_column_width=True)
