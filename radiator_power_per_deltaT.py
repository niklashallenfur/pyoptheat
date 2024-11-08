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

# start_time = datetime.fromisoformat("2024-10-19T14:00:00+02:00").timestamp()
start_time = datetime.fromisoformat("2024-10-21T14:00:00+02:00").timestamp()
end_time = datetime.fromisoformat("2024-10-22T12:00:00+02:00").timestamp()
# end_time = datetime.fromisoformat("2024-10-24T12:00:00+02:00").timestamp()

# Color mapping for each sensor
sensor_colors = {
    'sensor.radiator_flow_temp': "#FFAAAA",
    'sensor.radiator_return_temp': "#AAAAFF",
    'sensor.house_hall_temp': "#AAFFAA",
    'sensor.climate_coop_temperature': "#EEEE22",
    'sensor.acctank_laddeffekt2': "#333333"
}

# sensor_ entities = all keys of sensor_colors
sensor_entities = list(sensor_colors.keys())

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
AND last_updated_ts >= '{start_time}'
AND last_updated_ts <= '{end_time}'
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

# Define the minute timestamps within the time range
minute_timestamps = pd.date_range(start=datetime.fromtimestamp(start_time),
                                  end=datetime.fromtimestamp(end_time), freq='min')

df_interpolated = pd.DataFrame(index=minute_timestamps)

# Plot the data for each sensor

for sensor in sensor_entities:
    sensor_df = df[df['entity_id'] == sensor].set_index('last_updated_ts')['state']
    df_interpolated[sensor] = df_interpolated[sensor] = (sensor_df
                                                         .reindex(sensor_df.index.union(minute_timestamps))
                                                         .sort_index()
                                                         .interpolate(method='time'))

df_interpolated['radiator_temp_diff'] = df_interpolated['sensor.radiator_flow_temp'] - df_interpolated[
    'sensor.radiator_return_temp']
df_interpolated['acctank_laddeffekt2_slope'] = df_interpolated['sensor.acctank_laddeffekt2'].diff()
slope_threshold = 300  # Adjust this value as needed
df_interpolated['sensor.acctank_laddeffekt2'] = (df_interpolated.apply(
    lambda row: -row['sensor.acctank_laddeffekt2']
    if row['sensor.acctank_laddeffekt2'] < -1000 and abs(row['acctank_laddeffekt2_slope']) < slope_threshold
    else None,
    axis=1)
                                                 .rolling(window=60, min_periods=1).mean())

df_interpolated['power_temp_ratio'] = (df_interpolated['sensor.acctank_laddeffekt2']
                                       / df_interpolated['radiator_temp_diff'])
df_interpolated['power_temp_ratio'] = (df_interpolated['power_temp_ratio'].apply(
    lambda x: x if abs(x - 975) <= 975 else None).rolling(window=10, min_periods=1).median())

# Create a figure and a primary y-axis
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 15), sharex=True)

# Plot temperature data on the primary y-axis
for sensor in sensor_entities:
    if sensor != 'sensor.acctank_laddeffekt2':
        ax1.plot(df_interpolated.index, df_interpolated[sensor], label=sensor, color=sensor_colors[sensor])

# Create a secondary y-axis
ax2 = ax1.twinx()

# Plot power data on the secondary y-axis
ax2.plot(df_interpolated.index, df_interpolated['sensor.acctank_laddeffekt2'], label='sensor.acctank_laddeffekt2',
         color=sensor_colors['sensor.acctank_laddeffekt2'])

# Plot the difference on the primary y-axis
ax1.plot(df_interpolated.index, df_interpolated['radiator_temp_diff'], label='radiator_temp_diff', color='black')

# Configure the primary y-axis
ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature (°C)')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M:%S'))
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True)

# Configure the secondary y-axis
ax2.set_ylabel('Power (W)')
ax2.set_ylim(bottom=0)

# Combine legends from both y-axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

# Plot the power to temperature difference ratio on the second subplot
ax3.plot(df_interpolated.index, df_interpolated['power_temp_ratio'], label='Power/Temp Difference', color='purple')

# Configure the second subplot
ax3.set_ylabel('Power/Temp Difference (W/°C)')
ax3.set_xlabel('Time')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M:%S'))
ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True)
ax3.legend(loc='best')
ax3.set_title('Power to Temperature Difference Ratio over the last 12 hours')

plt.title('Temperature of Sensors and Radiator Temperature Difference over the last 12 hours')
plt.tight_layout()

# Show the plot
plt.show()
