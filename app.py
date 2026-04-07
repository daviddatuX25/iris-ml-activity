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
    # --- RADAR CHART VISUALIZATION LOGIC ---
    import plotly.graph_objects as go
    
    # Define the 4 axes of our radar chart
    categories = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    # We add the first value to the end of the list to "close" the shape loop
    values = [sepal_length, sepal_width, petal_length, petal_width]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(221, 160, 221, 0.6)', # Transparent purple/pink
        line=dict(color='#8fbc8f', width=3),  # Green border
        name='Current Measurements'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10]) # Lock the scale to 10cm max
        ),
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Render the interactive graph
    st.plotly_chart(fig_radar, use_container_width=True)

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
