import streamlit as st
import pandas as pd
import os
import joblib

# Streamlit app title
st.title('Fraud Detection Prediction App')

# File upload (assuming the user will upload a file now)
uploaded_file = st.file_uploader("Upload your Excel file (datos_futuros.xlsx)", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)

        st.subheader("Original Data:")
        st.write(df.head())

        # Data preprocessing steps (as in the original notebook)
        # Drop unn
