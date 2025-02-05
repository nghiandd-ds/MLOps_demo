import sqlite3
import os

# Define the folder where the database should be stored
db_folder = "data"  # Change this to your desired folder
db_path = os.path.join(db_folder, "data.db")

# Ensure the folder exists
os.makedirs(db_folder, exist_ok=True)

# Connect to SQLite database (creates file if it doesn't exist)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create a table to store new data from operation if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS accumulated_retrieval_data (
                LIMIT_BAL FLOAT,
                SEX FLOAT,
                EDUCATION FLOAT,
                MARRIAGE FLOAT,
                AGE FLOAT,
                PAY_1 FLOAT,
                PAY_2 FLOAT,
                PAY_3 FLOAT,
                PAY_4 FLOAT,
                PAY_5 FLOAT,
                PAY_6 FLOAT,
                BILL_AMT1 FLOAT,
                BILL_AMT2 FLOAT,
                BILL_AMT3 FLOAT,
                BILL_AMT4 FLOAT,
                BILL_AMT5 FLOAT,
                BILL_AMT6 FLOAT,
                PAY_AMT1 FLOAT,
                PAY_AMT2 FLOAT,
                PAY_AMT3 FLOAT,
                PAY_AMT4 FLOAT,
                PAY_AMT5 FLOAT,
                PAY_AMT6 FLOAT,
                default payment next month FLOAT,
                DATE6 FLOAT
);
""")

# Commit changes and close the connection
conn.commit()




