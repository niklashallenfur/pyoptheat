import pymysql
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Database connection details from .env file
db_host = os.getenv('DB_HOST')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')

# Define the sensor entities and time range
sensor_entities = [f'sensor.hsc_acc_p{i}' for i in range(1, 11)]
start_timestamp = datetime.fromisoformat("2024-10-19T10:00:00+02:00")
end_timestamp = datetime.fromisoformat("2024-10-20T17:00:00+02:00")

# Connect to the MariaDB database
conn = pymysql.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    database=db_name
)

# Prepare sensor entity IDs for the SQL IN clause
sensor_entity_ids = "','".join(sensor_entities)

# Query to get the data from the specified time range for all sensors
query = f"""
SELECT m.entity_id, s.state, s.last_updated_ts 
FROM states_meta m
JOIN states s ON m.metadata_id = s.metadata_id
WHERE m.entity_id IN ('{sensor_entity_ids}')
AND s.last_updated_ts BETWEEN UNIX_TIMESTAMP('{start_timestamp}') AND UNIX_TIMESTAMP('{end_timestamp}')
ORDER BY s.last_updated_ts ASC
"""

# Read data into a DataFrame
df = pd.read_sql(query, conn)

# Close the database connection
conn.close()

# Convert the last_updated_ts column to datetime
df['last_updated_ts'] = pd.to_datetime(df['last_updated_ts'], unit='s')

# Convert the state column to float (assuming temperature values are numeric)
df['state'] = pd.to_numeric(df['state'], errors='coerce')

# Drop any rows with NaN values
df.dropna(inplace=True)

# Pivot the data so that each sensor's temperature is in its own column
df_pivot = df.pivot_table(index='last_updated_ts', columns='entity_id', values='state')

# Resample the data to hourly intervals, taking the first value at the start of each hour
df_resampled = df_pivot.resample('H').first()

# Calculate the average temperature of all sensors at the start of each hour
df_resampled['average_temperature'] = df_resampled.mean(axis=1)

# Print the average temperature at the start of each hour
print(df_resampled[['average_temperature']])
