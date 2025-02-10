import sqlite3
import pandas as pd
import os
import joblib
from itertools import chain
import importlib.util
from datetime import datetime, timezone
import shutil
import mlflow

'''
Auto-training strategy:
- Trigger: when monitoring log return decision to re-train the model
- Step:
    + Load all data from 3 months prior to the nearest upto date of data used for making re-training decision
    + Use model-type, parameter, data pipeline of the established model
    + If model used scaled data, new scaler have to be make to fit with new train data
   
'''

# Function get path
def get_path(version, call_file):
    parent_folder = os.path.dirname(os.getcwd())
    for root, dirs, files in os.walk(parent_folder):
        if (call_file in files) and (version in root):
            path = root + "/" + call_file
            return(path)


# Connect to DB
conn = sqlite3.connect(get_path(version = "data", call_file = "data.db"))

# Select 3 months to retrain model
time_frame = pd.read_sql_query('select distinct TIME_FRAME from monitoring_result', conn)
text_list = time_frame['TIME_FRAME'].tolist()
format_date6 = lambda x: x.replace('{', '').replace('}', '').replace(' ', '').split(',')
chained_list = [int(l) for l in set(chain(*[format_date6(i) for i in text_list]))]
chained_list.sort()
data_year = str(chained_list[-4:-1]).replace('[', '(').replace(']', ')')

# Load data
data_query = 'SELECT * FROM accumulated_retrieval_data WHERE DATE6 in ' + data_year
df = pd.read_sql_query(data_query, conn)

# Load model info
model_info = joblib.load(get_path("streamlit-app","model_info.pkl"))

# Processing pipeline
pipeline_info_path = get_path(model_info["data_pipeline"], "pipeline_info.pkl")
pipeline_info = joblib.load(pipeline_info_path)

# Path to pipeline transform function
function_path = get_path(version = model_info['data_pipeline'], call_file = "pipeline.py")

# Load and run existed pipeline
spec = importlib.util.spec_from_file_location("pipeline", function_path)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

# Setup time UTC
now = int(datetime.now(timezone.utc).timestamp())

# Check for scaler exist and re-fit parameters
if pipeline_info['Scaler'] is not None:
    import_scaler = f"from {pipeline_info['Scaler'].__class__.__module__} import {pipeline_info['Scaler'].__class__.__name__}"
    exec(import_scaler)

    # Run update pipeline on new data
    updated_pipeline_info = pipeline.data_pipeline(df, pipeline_info_path).update_pipeline()

    # Make new folder for update pipeline
    new_folder_name = model_info["data_pipeline"] + "_re_" + str(now)
    new_folder_path = pipeline_info_path.replace(model_info["data_pipeline"] + "/" + "pipeline_info.pkl", new_folder_name)

    # Update all pipeline to new folder
    joblib.dump(updated_pipeline_info, new_folder_path + "/" + "pipeline_info.pkl") # update data pipeline information
    shutil.copy(get_path(version=model_info['data_pipeline'], call_file="pipeline.py"), new_folder_path) # Copy the pipeline code to new folder

    # Make a sample for future deployment tests
    sample_data = df.sample(100)
    validation_data = {
        'input_data': sample_data,
        'output_data': pipeline.data_pipeline(sample_data, new_folder_path + "/" + "pipeline_info.pkl").fit()
    }
    joblib.dump(validation_data, new_folder_path + "/" + "validation_data.pkl")

else:
    new_folder_name = model_info["data_pipeline"]
    new_folder_path = pipeline_info_path


# Prepare data for ra-train
X = pipeline.data_pipeline(df.drop('default payment next month', axis=1), new_folder_path + "/" + "pipeline_info.pkl").fit()
y = df['default payment next month']

# Load established model for model type and parameter
model_path = get_path(model_info['model'],"model.pkl")
established_model = joblib.load(model_path)
import_model = f"from {established_model.__class__.__module__} import {established_model.__class__.__name__} as model"
exec(import_model)

# Fit new model
new_model = model(**established_model.get_params())
new_model.fit(X, y)

# Make new folder for update pipeline
new_model_name = model_info["model"] + "_re_" + str(now)
new_model_path = model_path.replace(model_info["model"] + "/" + "model.pkl", new_model_name)

# Use MLflow to log new model
mlflow.sklearn.save_model(new_model, new_model_path, input_example=X.sample(frac=0.1))

# Make training log for future version control
training_log = pd.DataFrame({
    'OLD_VERSION' : [model_info],
    'NEW_VERSION' : [{'data_pipeline': new_folder_name, 'model': new_model_name}],
    'UPDATE_LOG' : ['Re-train at UTC ' + str(datetime.now(timezone.utc)) + ". Data used: " + data_year + "."],
    'PIPELINE_FOLDER' : [new_folder_path],
    'MODEL_FOLDER' : [new_model_path]
})

training_log.to_sql("training_log", conn, if_exists="append", index=False)

# Close the connection to DB
conn.close()


