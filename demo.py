# Script implementation of the demo for integration with Streamlit

import streamlit as st
import cv2
from visualization.visualizer import Visualizer
import numpy as np

# Define paths to the models
model_paths = {
    "MNIST": "./trained_models/mnist/",
    "Fashion MNIST": "./trained_models/fashion/",
    "dSprites": "./trained_models/dsprites/"
}

# Load visualizers for each model
visualizers = {name: Visualizer(path) for name, path in model_paths.items()}


# Define function to update image
def update_mnist_image(number_type, width, style1, style2, tilt, thickness):
    continuous = [width, style1, tilt, 0.0, 0.0, 0.0, thickness, 0.0, style2, 0.0]
    discrete = np.zeros(10, int)
    number_mapping = [2, 8, 9, 1, 6, 3, 5, 4, 7, 0]
    discrete[number_mapping.index(number_type)] = (
        3 # 3 for a good sharpness, fixed instead of ppf
    )
    img = visualizers['MNIST'].visualize(
        discrete_latent_parameters=discrete, continous_latent_parameters=continuous
    )
    return img

def update_fashion_mnist_image(size, diffusion, style, shape, garment_type):
    continuous = [0.0, size, 0.0, diffusion, 0.0, style, 0.0, 0.0, 0.0, shape]
    discrete = np.zeros(10, int)
    discrete[garment_type] = (
        3 # 3 for a good sharpness, fixed instead of ppf
    )
    img = visualizers['Fashion MNIST'].visualize(
        discrete_latent_parameters=discrete, continous_latent_parameters=continuous
    )
    return img

def update_dsprites_image(yaxis, xaxis, shape, angle, scale1, scale2):
    continuous = [0, yaxis, xaxis, 0, shape, angle]
    discrete = [scale2, scale1, 0]
    img = visualizers["dSprites"].visualize(
        discrete_latent_parameters=discrete, continous_latent_parameters=continuous
    )
    return img


# Streamlit UI
st.title("Disentangled Variational Autoencoder Demo")

# Dropdown menu for model selection
model_selection = st.sidebar.selectbox(
    "Select Model", 
    ("MNIST", "Fashion MNIST", "dSprites")
)

# Define sliders for each parameter based on selected model
if model_selection == "MNIST":
    number_type = st.sidebar.slider("Number type", 0, 9, 4)
    width = st.sidebar.slider("Width", -3.0, 3.0, 0.0, 0.1)
    style1 = st.sidebar.slider("Style 1", -3.0, 3.0, 0.0, 0.1)
    style2 = st.sidebar.slider("Style 2", -3.0, 3.0, 0.0, 0.1)
    tilt = st.sidebar.slider("Tilt", -3.0, 3.0, 0.0, 0.1)
    thickness = st.sidebar.slider("Thickness", -3.0, 3.0, 0.0, 0.1)
    img = update_mnist_image(number_type, width, style1, style2, tilt, thickness)

elif model_selection == "dSprites":
    yaxis = st.sidebar.slider("Y-Axis", -10.0, 10.0, 0.0, 1.0)
    xaxis = st.sidebar.slider("X-Axis", -10.0, 10.0, 0.0, 1.0)
    shape = st.sidebar.slider("Shape", -3.0, -1.0, -2.0, 1.0)
    angle = st.sidebar.slider("Angle", -4.0, 3.0, 0.0, 1.0)
    scale1 = st.sidebar.slider("Scale 1", 0, 10, 0, 1)
    scale2 = st.sidebar.slider("Scale 2", 0, 6, 0, 1)
    img = update_dsprites_image(yaxis, xaxis, shape, angle, scale1, scale2)

elif model_selection == "Fashion MNIST":
    garment_types = ["Dress", "Bag", "Ankle Boot", "Pullover", "Coat", "Trouser", "Shirt", "Sandale"]
    selected_garment = st.sidebar.select_slider("Garment Type", options=garment_types, value="Dress")
    attr1 = st.sidebar.slider("Size", -10.0, 2.0, 0.0, 1.)
    attr2 = st.sidebar.slider("Extravagance", -10.0, 10.0, 0.0, 1.)
    attr3 = st.sidebar.slider("Style", -10.0, 10.0, 0.0, 1.)
    attr4 = st.sidebar.slider("Cut", -10.0, 10.0, 0.0, 1.)
    selected_garment_index = garment_types.index(selected_garment)
    img = update_fashion_mnist_image(attr1, attr2, attr3, attr4, selected_garment_index)

# Using the Inter_Nearest interpolation for enhanced sharpness due to very low resolution.
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
st.image(img, caption="Generated Image", use_column_width=True)
