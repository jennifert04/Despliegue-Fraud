# prompt: haz el despliegue anterior pero en streamlit

import streamlit as st
import pandas as pd
import joblib
import os

# Set up the path to your files in Google Drive
# You'll need to ensure these files are accessible (e.g., shared or copied to a public folder)
# or you could upload them directly in the Streamlit app if they are small.
# For this example, let's assume they are in a location accessible to the app.
# A more robust solution for production would involve storing models/data in cloud storage
# and accessing them securely.

# For this example, we'll assume the files are in the same directory as the Streamlit script
# For a production environment, you'd likely want a more structured approach.
# scaler_filename = os.path.join('scaler_fraud.pkl')
# model_filename = os.path.join('best_fraud_detection_model_SVM.pkl')

# Load the scaler and the model
# @st.cache_resource # Use this if the model/scaler are large and don't change often
def load_resources():
    try:
        loaded_scaler = joblib.load('scaler_fraud.pkl')
        loaded_model = joblib.load('best_fraud_detection_model_SVM.pkl')
        return loaded_scaler, loaded_model
    except FileNotFoundError:
        st.error("Model or scaler files not found. Please ensure 'scaler_fraud.pkl' and 'best_fraud_detection_model_SVM.pkl' are in the same directory as the script.")
        return None, None

loaded_scaler, loaded_model = load_resources()

if loaded_scaler is not None and loaded_model is not None:
    st.title("Fraud Detection App")

    st.write("Upload a CSV or Excel file to predict fraud.")

    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("Original Data:")
            st.dataframe(df.head())

            # Preprocessing steps mirroring your notebook
            # Drop columns that are not needed for prediction
            columns_to_drop = ['newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud']
            for col in columns_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])
                else:
                    st.warning(f"Column '{col}' not found in the uploaded file.")

            # Convert 'type' column to category and then one-hot encode
            if 'type' in df.columns:
                if df['type'].dtype == 'object':
                    df['type'] = df['type'].astype('category')
                df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)
            else:
                st.warning("Column 'type' not found in the uploaded file.")


            # Apply the loaded scaler to the specified columns
            scaling_cols = ['amount', 'newbalanceOrig']
            for col in scaling_cols:
                if col not in df.columns:
                    st.warning(f"Column '{col}' not found for scaling.")

            # Ensure all necessary dummy columns for 'type' exist, add if not, fill with 0
            expected_type_cols = [f'type_{t}' for t in ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']] # Replace with actual categories if different
            for col in expected_type_cols:
                if col not in df.columns:
                    df[col] = 0

            # Ensure the order of columns matches the training data if necessary for the model
            # This is a crucial step if your model is sensitive to feature order.
            # You might need to load the list of columns used during training.
            # For this example, we'll assume the column names and order are consistent
            # after preprocessing and one-hot encoding.

            # Scale the relevant columns *after* one-hot encoding if the scaler was fitted on the combined data
            # If the scaler was fitted only on 'amount' and 'newbalanceOrig' before one-hot encoding,
            # apply it before. Let's assume it was fitted before one-hot encoding based on your notebook.
            # However, scaling after one-hot encoding is also common depending on the workflow.
            # Re-applying fit_transform might not be correct, as you should only transform on new data.
            # It's better to use `transform`.
            cols_to_scale_present = [col for col in scaling_cols if col in df.columns]
            if cols_to_scale_present:
                # Create a temporary DataFrame with just the columns to scale
                df_to_scale = df[cols_to_scale_present]

                # Transform these columns
                scaled_values = loaded_scaler.transform(df_to_scale)

                # Update the original DataFrame with the scaled values
                df[cols_to_scale_present] = scaled_values


            # Ensure all columns expected by the model are present.
            # A robust way is to store the list of training columns and realign.
            # Example (assuming you saved the training columns list):
            # training_columns = joblib.load('training_columns.pkl') # Load this file
            # df_processed = df.reindex(columns=training_columns, fill_value=0)

            # Make predictions
            try:
                # Ensure column order matches the model's expectations
                # This is a simplified approach; a better way is to store and use the list of training columns
                # and reindex the input dataframe.
                # For instance, if your model expects ['amount', 'newbalanceOrig', 'type_CASH_IN', ...],
                # ensure the dataframe `df` has these columns in the correct order.
                # Let's assume the current `df` after processing has the correct columns and order for prediction.
                predictions = loaded_model.predict(df)

                # Add predictions to the DataFrame
                df['predictions'] = predictions

                st.write("Predictions:")
                st.dataframe(df[['amount', 'newbalanceOrig', 'type_CASH_OUT', 'type_PAYMENT', 'type_CASH_IN', 'type_TRANSFER', 'type_DEBIT', 'predictions']].head()) # Displaying relevant columns

                st.write("Prediction meaning: 1 = Fraud, 0 = Not Fraud")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.write("Please check if the uploaded file's columns and data types match the expected format for the model.")

        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")

