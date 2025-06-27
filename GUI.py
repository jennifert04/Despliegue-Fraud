# prompt: haz el despliegue anterior pero en streamlit

import streamlit as st
import pandas as pd
import joblib
import os

# Title for the app
st.title('Fraud Detection Prediction App')

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_excel(uploaded_file)

    # Display the head of the dataframe
    st.subheader('Original Data')
    st.write(df.head())

    # --- Data Preprocessing ---
    # Drop specified columns if they exist
    cols_to_drop = ['newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Convert object columns to category
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    # Perform one-hot encoding for 'type' column
    if 'type' in df.columns:
        df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

    # --- Scaling ---
    # Assuming the scaler file is available in the environment where Streamlit is running
    scaler_filename = 'standard_scaler_fraud.pkl'
    if os.path.exists(scaler_filename):
        loaded_scaler = joblib.load(scaler_filename)
        # Check if 'amount' and 'newbalanceOrig' columns exist before scaling
        cols_to_scale = [col for col in ['amount', 'newbalanceOrig'] if col in df.columns]
        if cols_to_scale:
             df[cols_to_scale] = loaded_scaler.transform(df[cols_to_scale])
        else:
            st.warning("Columns 'amount' or 'newbalanceOrig' not found for scaling.")
    else:
        st.error(f"Scaler file '{scaler_filename}' not found. Please make sure it's in the same directory as your app.")
        st.stop() # Stop execution if scaler is not found

    # Display processed data head
    st.subheader('Processed Data (before prediction)')
    st.write(df.head())

    # --- Model Loading and Prediction ---
    # Assuming the model file is available in the environment where Streamlit is running
    model_filename = 'best_fraud_detection_model_SVM.pkl'
    if os.path.exists(model_filename):
        loaded_model = joblib.load(model_filename)

        # Make predictions
        try:
            predictions = loaded_model.predict(df)

            # Add predictions to the DataFrame
            df['predictions'] = predictions

            # Display results
            st.subheader('Predictions')
            st.write(df[['predictions']].head())
            st.write(f"Number of fraudulent transactions predicted: {df['predictions'].sum()}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.warning("Please ensure the columns in your uploaded file match the format expected by the model after preprocessing.")

    else:
        st.error(f"Model file '{model_filename}' not found. Please make sure it's in the same directory as your app.")
        st.stop() # Stop execution if model is not found

