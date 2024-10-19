import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()

# Database connection details from .env
db_host = os.getenv('DB_HOST')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')

# Define the sensor entities (hsc_acc_p1 to hsc_acc_p10) and the time range (last 12 hours)
sensor_entities = [f'sensor.hsc_acc_p{i}' for i in range(1, 11)]
time_12_hours_ago = time.time() - timedelta(hours=6).seconds

# Color mapping for each sensor
sensor_colors = {
    'sensor.hsc_acc_p1': "#ff0000",
    'sensor.hsc_acc_p2': "#ffa500",
    'sensor.hsc_acc_p3': "#ffff00",
    'sensor.hsc_acc_p4': "#7fff00",
    'sensor.hsc_acc_p5': "#00ff00",
    'sensor.hsc_acc_p6': "#00ffff",
    'sensor.hsc_acc_p7': "#007fff",
    'sensor.hsc_acc_p8': "#0000ff",
    'sensor.hsc_acc_p9': "#4b0082",
    'sensor.hsc_acc_p10': "#8f00ff",
}

# Connect to the MariaDB database
conn = pymysql.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    database=db_name
)

# Prepare sensor entity IDs for the SQL IN clause
sensor_entity_ids = "','".join(sensor_entities)

# Query to get the last 12 hours of sensor data for all sensors
query = f"""
SELECT m.entity_id, s.state, s.last_updated_ts 
FROM states_meta m
JOIN states s ON m.metadata_id = s.metadata_id
WHERE m.entity_id IN ('{sensor_entity_ids}')
AND last_updated_ts >= '{time_12_hours_ago}'
"""

print(query)

# Read data into a DataFrame
df = pd.read_sql(query, conn)

# Print the DataFrame to debug
print(df.head())  # Display the first few rows of the DataFrame
print(df.info())  # Display the data types and non-null count

# Close the database connection
conn.close()

# Convert the last_updated_ts column to datetime
df['last_updated_ts'] = pd.to_datetime(df['last_updated_ts'], unit='s')

# Convert the state column to float (assuming temperature values are numeric)
df['state'] = pd.to_numeric(df['state'], errors='coerce')

# Drop any rows with NaN values
df.dropna(inplace=True)

# Plot the data for each sensor
plt.figure(figsize=(10, 6))
for sensor in sensor_entities:
    sensor_df = df[df['entity_id'] == sensor]
    plt.plot(sensor_df['last_updated_ts'], sensor_df['state'], label=sensor, color=sensor_colors[sensor])

# Configure the plot
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Temperature of Sensors hsc_acc_p1 to hsc_acc_p10 over the last 12 hours')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()

# Show the plot
plt.show()
