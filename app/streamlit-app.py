import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import joblib
import os
from sklearn.preprocessing import StandardScaler

class data_pipeline():
    def data_transform_pipeline(self, data):
        data['Num_of_month_payment_delayed_1'] = data['PAY_1'].apply(self.transform_pay_to_delay_payment)
        data['Num_of_month_payment_delayed_2'] = data['PAY_2'].apply(self.transform_pay_to_delay_payment)
        data['Num_of_month_payment_delayed_3'] = data['PAY_3'].apply(self.transform_pay_to_delay_payment)
        data['Num_of_month_payment_delayed_4'] = data['PAY_4'].apply(self.transform_pay_to_delay_payment)
        data['Num_of_month_payment_delayed_5'] = data['PAY_5'].apply(self.transform_pay_to_delay_payment)
        data['Num_of_month_payment_delayed_6'] = data['PAY_6'].apply(self.transform_pay_to_delay_payment)

        data['is_pay_duly_1'] = data['PAY_1'].apply(self.transform_pay_to_pay_duly)
        data['is_pay_duly_2'] = data['PAY_2'].apply(self.transform_pay_to_pay_duly)
        data['is_pay_duly_3'] = data['PAY_3'].apply(self.transform_pay_to_pay_duly)
        data['is_pay_duly_4'] = data['PAY_4'].apply(self.transform_pay_to_pay_duly)
        data['is_pay_duly_5'] = data['PAY_5'].apply(self.transform_pay_to_pay_duly)
        data['is_pay_duly_6'] = data['PAY_6'].apply(self.transform_pay_to_pay_duly)

        data.loc[:, 'MALE'] = [1 if x == 1 else 0 for x in data['SEX']]

        data.loc[:, 'MARRIAGE_MARRIED'] = [1 if x == 1 else 0 for x in data['MARRIAGE']]
        data.loc[:, 'MARRIAGE_SINGLE'] = [1 if x == 2 else 0 for x in data['MARRIAGE']]
        data.loc[:, 'MARRIAGE_OTHER'] = [1 if x == 3 else 0 for x in data['MARRIAGE']]

        data.loc[:, 'EDU_GRAD'] = [1 if x == 1 else 0 for x in data['EDUCATION']]
        data.loc[:, 'EDU_UNIVER'] = [1 if x == 2 else 0 for x in data['EDUCATION']]
        data.loc[:, 'EDU_HIGH'] = [1 if x == 3 else 0 for x in data['EDUCATION']]
        data.loc[:, 'EDU_OTHER'] = [1 if x == 4 else 0 for x in data['EDUCATION']]

        data = data[data['AGE'] >= 18].copy().drop([
            'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'MARRIAGE', 'EDUCATION', 'SEX', 'DATE6'
        ], axis=1).reset_index(drop=True)

        return data

    def transform_pay_to_delay_payment(self, x):
        if x > 0:
            return x
        else:
            return 0

    def transform_pay_to_pay_duly(self, x):
        if x == -1:
            return 1
        else:
            return 0

    def __init__(self, data):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "pipeline_info.pkl")
        print(file_path)
        self.pipeline_info = joblib.load("pipeline_version/v1/pipeline_info.pkl")
        self.data = self.data_transform_pipeline(data=data)
        version_control = "Data pipeline version 1.0"

    def fit(self):
        num_error = 0
        for i in self.pipeline_info['Columns']:
            if i not in self.data.columns.to_list():
                print("Error: Column " + i + " is required!")
                num_error += 1

        if num_error == 0:
            X_data_col_names = [i + '_scaler' for i in self.data.columns.to_list()]
            X_data = self.pipeline_info['Scaler'].transform(self.data[self.pipeline_info['Columns']])
            X_data = pd.DataFrame(X_data).set_axis(X_data_col_names, axis=1)
            return X_data

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
        processed_data = data_pipeline(data).fit()

        # Load model and make predictions
        champion_model = joblib.load("/model_versions/champion_model/model.pkl")
        predictions = champion_model.predict(processed_data)

        # Display predictions
        st.write("Predictions:")
        st.dataframe(pd.DataFrame(predictions, columns=["Prediction"]).head())

if __name__ == "__main__":
    main()


