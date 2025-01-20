import numpy as np
import pandas as pd
import sklearn
import streamlit
import joblib
from pipeline_version.v1.pipeline import data_pipeline()

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
        pipeline_info = '../model_versions/champion_model/model.pkl'
        processed_data = data_pipeline(data)

        # Load model and make predictions
        champion_model = joblib.load("../model_versions/champion_model/model.pkl")
        predictions = champion_model.predict(processed_data).fit()

        # Display predictions
        st.write("Predictions:")
        st.dataframe(pd.DataFrame(predictions, columns=["Prediction"]).head())

if __name__ == "__main__":
    main()


