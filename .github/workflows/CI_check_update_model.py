import numpy as np
import pandas as pd
import sklearn
import joblib
import importlib.util
import os
from sklearn.preprocessing import StandardScaler

def get_path(version, call_file):
    parent_folder = os.path.dirname(os.getcwd())
    for root, dirs, files in os.walk(parent_folder):
        if (call_file in files) and (version in root):
            path = root + "/" + call_file
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
champion_model = joblib.load(model_info)


# Load validation data
validation_data = joblib.load(get_path(version = info['data_pipeline'], call_file = 'validation_data.pkl'))

# Test function
def test_data_pipeline():
    assert len(validation_data["input_data"]) > 0, "ERROR: No validation data found."
    assert len(validation_data["input_data"]) == len(validation_data["output_data"]), "ERROR: Input and output of validation data did not match."
    assert (data_pipeline.data_pipeline(validation_data["input_data"], data_info).fit() != validation_data["output_data"]).sum().sum() == 0, "ERROR: Data pipeline did not return expected result."

def test_data_for_model():
    feature_in_model = champion_model.feature_names_in_.tolist()
    pipeline = joblib.load(data_info)
    if pipeline['Scaler'] is None:
        feature_in_pipeline = pipeline['Columns']
    else:
        feature_in_pipeline = [i + '_scaler' for i in pipeline['Columns']]
    feature_in_model.sort()
    feature_in_pipeline.sort()
    assert feature_in_model == feature_in_pipeline, "ERROR: Data pipeline does not match with model."

def test_prediction():
    predict_proba = champion_model.predict_proba(data_pipeline.data_pipeline(validation_data["input_data"], data_info).fit())
    assert len(set(predict_proba)) > 1, "ERROR: Model only return 1 similar prediction for all observations."