import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as st 

os.chdir(r"C:\Training\Academy\Statistics (Python)\Datasets")
cars93 = pd.read_csv("Cars93.csv")

num_columns = ['Price', 'MPG.city', 'MPG.highway', 
               'Horsepower', 'EngineSize', 'Weight']
cat_columns = [ None, 'Type', 'AirBags', 'Origin',
               'DriveTrain', 'Cylinders']

st.title("Scatter Plot for Cars93 Dataset")

col1, col2 = st.columns([0.25, 0.75])

with col1:
    x_axis = st.selectbox('X-Axis:', 
                          num_columns, index=0)
    y_axis = st.selectbox('Y-Axis:', 
                          num_columns, index=1)
    c_axis = st.selectbox('Colour:', cat_columns)
 
with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_axis, y=y_axis, hue=c_axis,
                    data=cars93)
    st.pyplot(fig)

