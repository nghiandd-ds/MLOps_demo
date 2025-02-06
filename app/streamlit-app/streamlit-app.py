import sqlite3
import pandas as pd
import sklearn
import streamlit as st
import joblib
import importlib.util
import os
from sklearn.preprocessing import StandardScaler
import json
import subprocess
import datetime

# Function to get path of file
def get_path(version, call_file):
    parent_folder = os.path.dirname(os.getcwd())
    for root, dirs, files in os.walk(parent_folder):
        if (call_file in files) and (version in root):
            path = root + "/" + call_file
            return(path)
def load_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return pd.DataFrame(data['data'], columns=data['columns'])

# Get data pipeline and model information
info = joblib.load(get_path("streamlit-app","model_info.pkl"))


# Load and run pipeline
spec = importlib.util.spec_from_file_location("data_pipeline",
                                              get_path(version = info['data_pipeline'],
                                                       call_file = "pipeline.py"))
data_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_pipeline)

# Load and run monitoring test
tester = importlib.util.spec_from_file_location("monitoring",
                                              get_path(version = "function",
                                                       call_file = "monitoring_test.py"))
monitoring = importlib.util.module_from_spec(tester)
tester.loader.exec_module(monitoring)

# Load data pipeline info
data_info = get_path(version = info['data_pipeline'], call_file = "pipeline_info.pkl")


# Load model info
model_info = get_path(version = info['model'], call_file = "model.pkl")
champion_model = joblib.load(model_info)
st.write(model_info)
update_time = os.path.getmtime(model_info)
st.write(update_time)
st.write(datetime.datetime.fromtimestamp(update_time))

# Sort input columns
input_data_order = joblib.load(data_info)['Input_data']
input_data_columns = input_data_order.copy()
input_data_columns.sort()

# Test connect to database
def check_sqlite_connection():
    db_path = get_path(version = "data", call_file = "data.db")
    try:
        conn = sqlite3.connect(db_path)  # Connect to SQLite database
        status = "Success"
        conn.close()  # Close connection
    except sqlite3.Error as err:
        status = "Failed"
    assert status == 'Success', "Can't connect to database."

# Test load data pipeline
def test_load_data_pipeline():
    if hasattr(data_pipeline, "data_pipeline") and callable(getattr(data_pipeline, "data_pipeline")):
        check_load = 'Success'
    else:
        check_load = 'Failed'
    assert data_info is not None, "ERROR: Fail to load data pipeline."
    assert model_info is not None, "ERROR: Fail to load data pipeline."
    assert check_load == 'Success', "ERROR: Fail to load data pipeline."

# Test load data pipeline
def test_load_monitoring_test():
    if hasattr(monitoring, "calculate_csi") and callable(getattr(monitoring, "calculate_csi")):
        check_csi = 'Success'
    else:
        check_csi = 'Failed'
    if hasattr(monitoring, "ar") and callable(getattr(monitoring, "ar")):
        check_ar = 'Success'
    else:
        check_ar = 'Failed'

    assert check_csi == 'Success', "ERROR: Fail to load CSI test."
    assert check_ar == 'Success', "ERROR: Fail to load AR test."



def main():
    st.title("CSV Prediction App")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)
        upload_time = datetime.datetime.now()
        columns_in_file = data.columns.tolist().copy()
        columns_in_file.sort()
        y_label = 'default payment next month'

        if all(elem in columns_in_file for elem in input_data_columns):
            # Preprocess the data
            processed_data = data_pipeline.data_pipeline(data=data[input_data_order], file_path=data_info).fit()

            # Load model and make predictions
            predictions = champion_model.predict_proba(processed_data[champion_model.feature_names_in_.tolist()])
            # Case 1: uploaded data doesn't have y_label
            if y_label not in columns_in_file:
                # Display predictions
                st.write("Predictions:")
                st.write("Click the button below to download the CSV file:")
                data['Predictions'] = predictions.T[1]
                st.download_button(
                    label="Download prediction",
                    data=data.to_csv(index=False),
                    file_name="prediction.csv",
                    mime="text/csv"
                )
            # Case 2: uploaded data have y_label
            else:
                # Upload to DB
                DB_PATH = get_path(version = "data", call_file = "data.db")
                conn = sqlite3.connect(DB_PATH)
                data_to_save = data.copy()
                data_to_save['UPLOADED_TIME'] = upload_time
                data_to_save.to_sql("accumulated_retrieval_data", conn,
                            if_exists="append", index=False)


                conn.close()
                # Push to Github
                GITHUB_USERNAME = "nghiandd-ds"
                GITHUB_REPO = "MLOps_demo"
                GITHUB_TOKEN = st.secrets["github"]["token"] # Token saved in streamlit cloud
                GITHUB_URL = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"

                # Set up Git user
                subprocess.run(["git", "config", "--global", "user.email", "github-actions@github.com"])
                subprocess.run(["git", "config", "--global", "user.name", "github-actions[bot]"])
                subprocess.run(["git", "remote", "set-url", "origin", GITHUB_URL], check=True)

                # Add, commit, and push changes
                subprocess.run(["git", "add", DB_PATH], check=True)
                subprocess.run(["git", "commit", "-m", "Update SQLite DB"], check=True)  # Fixed commit message
                subprocess.run(["git", "push", "origin"], check=True)

                # Load model artifact (input data)
                model_artifact = load_json(get_path(version = 'champion_model',
                                                    call_file = 'input_example.json'))
                variable = []
                for i in champion_model.feature_names_in_:
                    csi = monitoring.calculate_csi(baseline = model_artifact[i],
                                                   current = processed_data[i])
                    ar = monitoring.ar(Y = data[y_label], X = processed_data[i])
                    variable.append([i, ar, csi])

                variable = pd.DataFrame(variable, columns = ["Variable", "AR", "CSI"])
                st.write("Monitoring result:")
                model_monitor = pd.DataFrame({
                    'MODEL': [f"{champion_model.__class__.__module__}.{champion_model.__class__.__name__}"],
                    'PARAMETER': [str(champion_model.get_params())],
                    'LAST_UPDATE': [],
                    'MONITORING_TIME': [upload_time],
                    'AR' : [monitoring.ar(Y = data[y_label], X = predictions.T[1])]
                })
                st.dataframe(model_monitor.T)
                st.write("Monitoring result:")
                st.write("By variables: ")
                st.dataframe(variable)

        else:
            st.write("Wrong data. Please make sure required columns are in the dataset")
            st.dataframe(pd.DataFrame({'Columns' : columns_in_file}))

if __name__ == "__main__":
    main()


