import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import joblib
import os
from sklearn.preprocessing import StandardScaler


# Function to get path of file
def get_path(version, call_file):
    parent_folder = os.path.dirname(os.getcwd())
    for root, dirs, files in os.walk(parent_folder):
        if (call_file in files) and (version in root):
            path = root + "\\" + call_file
            return(path)

# Get data pipeline and model information
info = joblib.load(get_path("streamlit-app","model_info.pkl"))


# Load and run pipeline
spec = importlib.util.spec_from_file_location("data_pipeline",
                                              get_path(version = info['data_pipeline'],
                                                       call_file = "pipeline.py"))
data_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_pipeline)


# Load data pipeline info
data_info = get_path(version = info['data_pipeline'], call_file = "pipeline_info.pkl")

# Load model info
model_info = get_path(version = info['model'], call_file = "model.pkl")


def main():
    st.title("CSV Prediction App")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(data.head())

        # Preprocess the data
        st.write("Preprocessing Data...")
        processed_data = data_pipeline.data_pipeline(data, data_info).fit()

        # Load model and make predictions
        champion_model = joblib.load(model_info)
        predictions = champion_model.predict(processed_data)

        # Display predictions
        st.write("Predictions:")
        st.dataframe(pd.DataFrame(predictions, columns=["Prediction"]).head())

if __name__ == "__main__":
    main()


