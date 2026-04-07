import streamlit as st
import pickle
import numpy as np

# Set up the page layout and title
st.set_page_config(page_title="Iris Predictor | BSIT", page_icon="🌸", layout="centered")

# Load the exported model efficiently
@st.cache_resource
def load_model():
    with open('knn_iris_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# App Header
st.title("🌸 Iris Species Predictor")
st.markdown("**Developed by: David Datu N. Sarmiento | ISPSC BSIT**")
st.markdown("Adjust the measurements below. The K-Nearest Neighbors (KNN) model will predict the specific Iris species in real-time.")
st.divider()

# Input Form
st.subheader("Flower Measurements (cm)")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.5, step=0.1)

with col2:
    sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.2, step=0.1)
    petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.4, step=0.1)

st.divider()

# Prediction Logic
if st.button("Predict Species 🚀", use_container_width=True, type="primary"):
    
    # Format inputs for the model
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Get prediction
    prediction = model.predict(features)
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    predicted_species = species_names[prediction[0]]
    
    # Display Result
    st.success(f"### Result: **Iris {predicted_species}**")
