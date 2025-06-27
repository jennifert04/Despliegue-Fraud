import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np # Import numpy for NaN and inf checks
import sklearn # Import sklearn to check version

# Streamlit app title
st.title('Fraud Detection Prediction App')

# File upload
uploaded_file = st.file_uploader("Upload your Excel file (e.g., datos_futuros.xlsx)", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)

        st.subheader("Original Data:")
        st.write(df.head())

        # --- Data preprocessing steps ---
        # Define columns to drop.
        # Based on your model's reported training features (7 columns, no oldbalanceOrg),
        # we are explicitly dropping 'oldbalanceOrg' from the input data.
        columns_to_drop = [
            'newbalanceDest', 'oldbalanceDest', 'step',
            'nameOrig', 'nameDest', 'isFlaggedFraud',
            'oldbalanceOrg' # Explicitly dropping oldbalanceOrg to match reported training data
        ]
        # Drop only the columns that are present in the current DataFrame
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # Convert object columns to category (assuming 'type' is the primary object column)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')

        # One-hot encode 'type' column
        df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

        # Define the paths to the scaler and model files
        scaler_filename = 'standard_scaler_fraud.pkl'
        model_filename = 'best_fraud_detection_model_SVM.pkl' # Ensure this matches your model file name

        # Check if the scaler and model files exist
        if not os.path.exists(scaler_filename):
            st.error(f"Error: Scaler file '{scaler_filename}' not found. Please ensure it's in the correct directory.")
            st.stop()
        if not os.path.exists(model_filename):
            st.error(f"Error: Model file '{model_filename}' not found. Please ensure it's in the correct directory.")
            st.stop()

        # Load the scaler and model
        try:
            loaded_scaler = joblib.load(scaler_filename)
            loaded_model = joblib.load(model_filename)
            st.success("Scaler and Model loaded successfully!")
            st.info(f"Tipo de scaler cargado: {type(loaded_scaler)}") # Display type of loaded scaler
        except Exception as e:
            st.error(f"Error loading scaler or model: {e}")
            st.stop()

        # Define columns to be scaled. 'oldbalanceOrg' is *NOT* in this list,
        # as it's being dropped and wasn't part of the reported training features.
        cols_to_scale = ['amount', 'newbalanceOrig']
        present_cols_to_scale = [col for col in cols_to_scale if col in df.columns]

        if present_cols_to_scale:
            st.subheader(f"Valores originales de las columnas a escalar ({present_cols_to_scale}):")
            st.write(df[present_cols_to_scale].head()) # Display values before scaling

            df[present_cols_to_scale] = loaded_scaler.transform(df[present_cols_to_scale])
            st.info(f"Columnas escaladas: {present_cols_to_scale}")
        else:
            st.warning("No se encontraron columnas para escalar entre 'amount' y 'newbalanceOrig'.")
            st.warning("Esto podría causar problemas si su modelo espera que estas características estén escaladas.")

        # Display DataFrame head *immediately after scaling* to verify
        st.subheader("DataFrame después de escalar:")
        st.write(df.head())


        # IMPORTANT: Ensure that the dataframe has all expected columns in the correct order.
        # This list NOW EXACTLY MATCHES your provided training DataFrame structure (7 columns).
        expected_columns = [
            'amount',
            'newbalanceOrig',
            'type_CASH_IN',
            'type_CASH_OUT',
            'type_DEBIT',
            'type_PAYMENT',
            'type_TRANSFER'
        ]

        # Add missing columns with default 0 to match the training data's feature set
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
                st.warning(f"Se añadió la columna faltante '{col}' con el valor predeterminado 0.")

        # Reorder columns to match the exact order of features used during model training
        df = df[expected_columns]

        # Show the preprocessed DataFrame before prediction
        st.subheader("DataFrame preprocesado final para predicción:")
        st.write(df.head())
        st.write(f"Dimensiones del DataFrame antes de la predicción: {df.shape}")
        st.write("Columnas y su orden antes de la predicción:")
        st.write(df.columns.tolist())

        # --- Debugging checks before prediction ---
        st.subheader("Comprobaciones de datos antes de la predicción:")
        # Check for NaN values
        if df.isnull().sum().sum() > 0:
            st.error("Advertencia: ¡El DataFrame contiene valores NaN después del preprocesamiento!")
            st.write(df.isnull().sum()) # Show count of NaNs per column

        # Check for infinite values
        if np.isinf(df).sum().sum() > 0:
            st.error("Advertencia: ¡El DataFrame contiene valores infinitos después del preprocesamiento!")
            st.write(np.isinf(df).sum()) # Show count of Infs per column

        # Display data types
        st.write("Tipos de datos de las columnas del DataFrame preprocesado:")
        st.write(df.dtypes)

        # Display scikit-learn version
        st.info(f"Versión de Scikit-learn: {sklearn.__version__}")


        # Make predictions
        try:
            predictions = loaded_model.predict(df)

            # Add predictions to the DataFrame
            df['isFraud_prediction'] = predictions

            st.subheader("Datos con Predicciones:")
            st.write(df.head())

            st.subheader("Resumen de Predicción:")
            fraud_count = df['isFraud_prediction'].sum()
            total_count = len(df)
            st.write(f"Registros totales: {total_count}")
            st.write(f"Transacciones fraudulentas predichas: {fraud_count}")
            st.write(f"Transacciones legítimas predichas: {total_count - fraud_count}")


        except Exception as e:
            st.error(f"Error durante la predicción: {e}")
            st.write("Ocurrió un error de predicción. Esto probablemente se deba a una falta de coincidencia en la versión de scikit-learn entre cuando el modelo fue guardado y cuando está siendo cargado/usado, o un problema con los datos mismos (ej. NaNs, Infs).")
            st.write(f"Versión actual de scikit-learn: {sklearn.__version__}")
            st.write(f"Columnas esperadas por el modelo (de la lista `expected_columns`): {expected_columns}")
            st.write(f"Columnas en el DataFrame preprocesado: {df.columns.tolist()}")
            st.write("Por favor, verifique dos veces la versión de scikit-learn utilizada durante el entrenamiento del modelo y asegúrese de que coincida exactamente con su entorno actual. Si la versión coincide y el error persiste, considere guardar nuevamente su modelo con la versión actual de scikit-learn.")

    except Exception as e:
        st.error(f"Error procesando el archivo cargado: {e}")
        st.write("Asegúrese de que el archivo sea un archivo Excel válido y que su contenido coincida con el formato esperado (ej. nombres de columnas).")

