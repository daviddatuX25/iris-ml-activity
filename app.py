import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn import datasets

# Set up the page layout and title
st.set_page_config(page_title="Iris Predictor | BSIT", page_icon="🌸", layout="wide") 

# Load the exported model efficiently
@st.cache_resource
def load_model():
    with open('knn_iris_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# App Header
st.title("🌸 Interactive Iris Predictor")
st.markdown("""
**Developed by:**
1. David Datu Sarmiento
2. Aljhun Angala
3. Christine Lopez
4. Allyza Kaye Wong
5. Charlene Hipolito

**ISPSC-TAGUDIN BSIT-3A**
""")


st.markdown("🐙 **[View our Source Code and Dataset on GitHub](https://github.com/daviddatux25/iris-ml-activity)**")

st.divider()



# Create two columns: Left for Inputs, Right for the Live Visualization
col_input, col_viz = st.columns([1, 1])

with col_input:
    st.subheader("1. Adjust Measurements (cm)")
    st.markdown("Change the values below and watch the flower on the right change its shape!")
    
    # Inputs
    sepal_length = st.number_input("Sepal Length (Green, Top/Bottom)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (Green, Top/Bottom)", min_value=1.0, max_value=10.0, value=3.2, step=0.1)
    petal_length = st.number_input("Petal Length (Purple, Diagonal)", min_value=1.0, max_value=10.0, value=1.5, step=0.1)
    petal_width = st.number_input("Petal Width (Purple, Diagonal)", min_value=0.1, max_value=10.0, value=0.4, step=0.1)

with col_viz:
    st.subheader("2. Live Flower Visualization")
    
    # --- SVG GENERATION LOGIC ---
    sl = sepal_length * 30
    sw = sepal_width * 30
    pl = petal_length * 30
    pw = petal_width * 30
    
    # SVG string with zero indentation to prevent Markdown code block formatting
    svg_code = f"""<div style="display: flex; justify-content: center; align-items: center; height: 100%; background-color: #f0f2f6; border-radius: 10px; padding: 20px;">
<svg width="300" height="300" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
<ellipse cx="150" cy="150" rx="{sw/2}" ry="{sl/2}" fill="#8fbc8f" opacity="0.8" />
<ellipse cx="150" cy="150" rx="{sw/2}" ry="{sl/2}" fill="#8fbc8f" opacity="0.8" transform="rotate(90 150 150)" />
<ellipse cx="150" cy="150" rx="{pw/2}" ry="{pl/2}" fill="#dda0dd" opacity="0.9" transform="rotate(45 150 150)" />
<ellipse cx="150" cy="150" rx="{pw/2}" ry="{pl/2}" fill="#dda0dd" opacity="0.9" transform="rotate(135 150 150)" />
<circle cx="150" cy="150" r="8" fill="#ffb6c1" />
</svg>
</div>"""
    
    # Render the SVG in Streamlit
    st.markdown(svg_code, unsafe_allow_html=True)

st.divider()

# Prediction Logic at the bottom
if st.button("Predict Species with KNN 🚀", use_container_width=True, type="primary"):
    
    # Format inputs for the model
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Get prediction
    prediction = model.predict(features)
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    predicted_species = species_names[prediction[0]]
    
    # Display Result
    st.success(f"### 🤖 The Model Predicts: **Iris {predicted_species}**")
    
    # Provide a fun fact based on the prediction
    if predicted_species == 'Setosa':
        st.info("Notice how tiny the purple petals are in the visualization? That is the classic signature of an Iris Setosa!")
    elif predicted_species == 'Versicolor':
        st.info("The petals and sepals are fairly balanced, a common trait of the Versicolor species.")
    else:
        st.info("Notice how large both the petals and sepals are? Virginica is generally the largest of the three species.")
        
    st.divider()
    
    # --- 3D GRAPH GENERATION LOGIC ---
    st.subheader("3. 3D Nearest Neighbors Analysis")
    st.markdown(f"See where your **{predicted_species}** flower lands compared to the original dataset. You can click, drag, and zoom this 3D graph!")
    
    # Load the background dataset for the graph
    iris_data = datasets.load_iris()
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['Species'] = [iris_data.target_names[i].capitalize() for i in iris_data.target]
    
    # Create the 3D Scatter Plot
    fig = px.scatter_3d(
        df, 
        x='sepal length (cm)', 
        y='sepal width (cm)', 
        z='petal length (cm)',
        color='Species', 
        color_discrete_sequence=['#ef553b', '#00cc96', '#ab63fa'],
        opacity=0.5
    )
    
    # Add the User's Custom Flower as a giant marker
    fig.add_trace(go.Scatter3d(
        x=[sepal_length], 
        y=[sepal_width], 
        z=[petal_length],
        mode='markers',
        marker=dict(size=15, color='black', symbol='diamond'),
        name='🌸 YOUR INPUT'
    ))
    
    # Clean up the layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
    
    # Render it in Streamlit
    st.plotly_chart(fig, use_container_width=True)
