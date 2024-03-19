import streamlit as st 
from joblib import load 
import pandas as pd 

#### Deserialization
#obj_load = load("stack-best.joblib")
obj_load = load("knn.joblib")
st.title("Inferencing on Concrete Mixtures")

col1, col2, col3 = st.columns([0.3, 0.3, 0.3], gap="small")

with col1:
    cement = st.slider("Cement:", min_value=100, 
                       max_value=500, value=400, step=1)
    blast = st.slider("Blast:",min_value=0, 
                      max_value=400, value=200, step=1)
    fly = st.slider("Fly:",min_value=0, 
                    max_value=250, value=100, step=1)
with col2:
    coarse =  st.slider("Coarse:",min_value=800, 
                        max_value=1200, value=900, step=1)
    water = st.slider("Water:", min_value=100, 
                      max_value=250, value=200, step=1)
    superPlast = st.slider("Superplasticizer:",
                           min_value=0, max_value=40, value=20, step=1)
with col3:
    fine = st.slider("Fine:",min_value=500, 
                     max_value=1000, value=600, step=1)
    age = st.number_input("Age:",min_value=1, 
                          max_value=365, value=100, step=1)
    df = pd.DataFrame({'Cement':[cement], 'Blast':[blast],
                      'Fly':[fly], 'Water':[water], 
                      'Superplasticizer':[superPlast],
                      'Coarse':[coarse], 'Fine':[fine],
                      'Age':[age]})
    pred = obj_load.predict(df)[0]
    st.write("Predicted Strength:{}" .format(pred) )
