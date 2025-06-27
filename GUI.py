# prompt: haz todo el depliegue anterior en streamlit

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import joblib
import tempfile

# Streamlit app title
st.title("Fraud Detection Prediction")

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        # Use a temporary file to read the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        df = pd.read_excel(tmp_file_path)
        os.remove(tmp_file_path)  # Remove the temporary file
        return df
    return None

# Function to preprocess the data
def preprocess_data(df, scaler_path):
    if df is None:
        return None

    df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg'])
    df = df.drop(columns=['step', 'nameOrig', 'nameDest'])
    df = df.drop(columns=['isFlaggedFraud'])

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

    # Load and apply the scaler
    if os.path.exists(scaler_path):
        loaded_scaler = joblib.load(scaler_path)
        # Ensure only existing columns are transformed
        cols_to_scale = ['amount', 'newbalanceOrig']
        cols_to_scale_exist = [col for col in cols_to_scale if col in df.columns]
        if cols_to_scale_exist:
             df[cols_to_scale_exist] = loaded_scaler.transform(df[cols_to_scale_exist])
    else:
        st.error(f"Scaler file not found at: {scaler_path}")
        return None

    return df

# Function to load the model and make predictions
def predict_fraud(df, model_path):
    if df is None:
        return None

    if os.path.exists(model_path):
        loaded_model = joblib.load(model_path)

        # Ensure the columns of the input DataFrame match the columns the model was trained on
        # This is a common issue when deploying models. You might need to save the training columns
        # and reindex the input data accordingly. For simplicity here, we assume column order matches.
        # A more robust approach would involve ensuring column names and order are identical.

        try:
            predictions = loaded_model.predict(df)
            df['predictions'] = predictions
            return df
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.warning("Please ensure the input data columns match the training data columns.")
            return None
    else:
        st.error(f"Model file not found at: {model_path}")
        return None


# File upload
uploaded_file = st.file_uploader("Upload your Excel file (datos_futuros.xlsx)", type=["xlsx"])

# Define model and scaler paths (adjust these paths based on your deployment structure)
# If running locally, these should be local paths. If deploying to cloud,
# consider using relative paths or environment variables.
# Example: If scaler and model are in the same directory as the streamlit app.py file:
scaler_path = 'standard_scaler.pkl'
model_path = 'best_fraud_detection_model_Naive_Bayes.pkl'


if uploaded_file is not None:
    st.write("File uploaded successfully. Loading and processing data...")

    df_original = load_data(uploaded_file)

    if df_original is not None:
        st.write("Original Data:")
        st.dataframe(df_original.head())

        st.write("Preprocessing data...")
        df_processed = preprocess_data(df_original.copy(), scaler_path)

        if df_processed is not None:
            st.write("Making predictions...")
            df_with_predictions = predict_fraud(df_processed.copy(), model_path)

            if df_with_predictions is not None:
                st.write("Predictions:")
                # Display the original data with predictions added
                st.dataframe(df_with_predictions)

                # Optional: Filter and display only the predicted frauds
                fraud_predictions = df_with_predictions[df_with_predictions['predictions'] == 1]
                if not fraud_predictions.empty:
                    st.write("Potential Fraud Transactions:")
                    st.dataframe(fraud_predictions)
                else:
                    st.write("No potential fraud transactions detected.")
