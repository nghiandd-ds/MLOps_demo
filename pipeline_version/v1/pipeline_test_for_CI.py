import importlib.util
import os
import joblib

def get_path(version, call_file):
    parent_folder = os.path.dirname(os.getcwd())
    for root, dirs, files in os.walk(parent_folder):
        if (call_file in files) and (version in root):
            path = root + "/" + call_file
            return(path)

# pre-set version for test
version = 'v1'

# Load function
spec = importlib.util.spec_from_file_location("data_pipeline",
                                              get_path(version = version,
                                                       call_file = "pipeline.py"))
data_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_pipeline)

# Test load file
def test_loading():
    if hasattr(data_pipeline, "data_pipeline") and callable(getattr(data_pipeline, "data_pipeline")):
        check_load = 'Success'
    else:
        check_load = 'Failed'
    assert (check_load == 'Success')


# Load data kiểm thử
validation_data = joblib.load('validation_data.pkl')
info = get_path(version, call_file = "pipeline_info.pkl")

# Test function
def test_pipeline():
    assert (len(validation_data["input_data"]) > 0)
    assert (len(validation_data["input_data"]) == len(validation_data["output_data"]))
    assert ((data_pipeline.data_pipeline(validation_data["input_data"], info).fit() != validation_data["output_data"]).sum().sum() == 0)
