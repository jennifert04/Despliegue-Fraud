import streamlit as st
import pandas as pd
import joblib
import os

# Function to preprocess the data
def preprocess_data(df, loaded_scaler):
    df['type'] = df['type'].astype('category')
    df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])


    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)
    df[['amount', 'newbalanceOrig']] = loaded_scaler.transform(df[['amount', 'newbalanceOrig']])
   
    return df

# Load the pre-trained models and transformers
# Assuming your models and transformers are in a directory named 'saved_models'
# in the same directory as your Streamlit app or accessible via a path.
try:
    scaler_filename = 'standard_scaler.pkl'
    loaded_scaler = joblib.load(scaler_filename)

    model_filename = 'best_fraud_detection_model_SVM.pkl'
    loaded_model = joblib.load(model_filename)
except FileNotFoundError:
    st.error("Error: Model file not found. Make sure files required exists.")
    st.stop()

# Streamlit App Title
st.title("Predicción de transacción fraudulenta")

st.write("""
Esta aplicación predice si una transacción es fraudulenta basado en los datos proporcionados.
""")

# File uploader for the user to upload their data
uploaded_file = st.file_uploader("Sube tu archivo Excel (solo .xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded Excel file into a pandas DataFrame
        df = pd.read_excel(uploaded_file)

        st.subheader("Datos cargados:")
        st.write(df.head())

        # Preprocess the data
        st.subheader("Datos preprocesados:")
        processed_df = preprocess_data(df.copy(), loaded_scaler)
        st.write(processed_df.head())

        if cols_to_scale_present:
            # Ensure the columns exist before attempting to scale
            df_processed[cols_to_scale_present] = loaded_scaler.transform(df_processed[cols_to_scale_present])
            st.write("Data after scaling:")
            st.dataframe(df_processed.head())
        else:
            st.warning(f"None of the specified columns for scaling ({numerical_cols_to_scale}) were found in the data.")


        # Align columns between the processed DataFrame and the model's training data
        # This is a critical step for deployment
        # Assuming the model was trained on a specific set of columns (including dummy variables)
        # We need to ensure the input data to the model has the same columns in the same order.

        # A robust approach is to save the list of columns from the training data
        # For this example, let's assume the columns needed are those after preprocessing
        # including the type dummy variables.
        # In a real-world scenario, you would save the `df.columns` after preprocessing your training data.

        # Dummy columns expected by the model (example - replace with actual columns from training)
        expected_cols_after_dummy = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'] # This list needs to match the training data columns

        # Create an empty DataFrame with the expected columns
        df_aligned = pd.DataFrame(0, index=df_processed.index, columns=expected_cols_after_dummy)

        # Copy the values from the processed DataFrame to the aligned DataFrame
        for col in df_aligned.columns:
            if col in df_processed.columns:
                df_aligned[col] = df_processed[col]

        st.write("Data after aligning columns for prediction:")
        st.dataframe(df_aligned.head())

        # Make predictions
        predictions = loaded_model.predict(df_aligned)

        # Add predictions to the original DataFrame (or a copy)
        df['Fraud Prediction'] = predictions

        st.subheader("Prediction Results:")
        st.write("0: Not Fraud, 1: Potential Fraud")
        st.dataframe(df[['amount', 'type', 'Fraud Prediction']].head())

        # Optional: Show rows predicted as fraud
        st.subheader("Transactions Predicted as Fraud:")
        fraudulent_transactions = df[df['Fraud Prediction'] == 1]
        if not fraudulent_transactions.empty:
            st.dataframe(fraudulent_transactions)
        else:
            st.write("No transactions predicted as fraud.")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
