import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as st 

dia = pd.read_csv("diamonds.csv")

num_columns = list(dia.dtypes[dia.dtypes != object].index)
cat_columns = [None]+list(dia.dtypes[dia.dtypes == object].index)

st.title("Box Plot")
col1, col2 = st.columns([0.25, 0.75])

with col1:
    n_col = st.selectbox('Measure:', 
                          num_columns, index=0)
    c_col = st.selectbox('Catogory:', 
                          cat_columns, index=0)
with col2:
    fig, ax = plt.subplots()
    sns.boxplot(x=c_col, y=n_col, data=dia)
    st.pyplot(fig)

