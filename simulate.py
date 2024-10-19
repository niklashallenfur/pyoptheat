import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
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

# Define the sensor entities (hsc_acc_p1 to hsc_acc_p10) and the time range (last 6 hours)
sensor_entities = [f'sensor.hsc_acc_p{i}' for i in range(1, 11)]
time_6_hours_ago = time.time() - timedelta(hours=6).total_seconds()

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

# Query to get the data from the last 6 hours for all sensors
query = f"""
SELECT m.entity_id, s.state, s.last_updated_ts 
FROM states_meta m
JOIN states s ON m.metadata_id = s.metadata_id
WHERE m.entity_id IN ('{sensor_entity_ids}')
AND last_updated_ts >= '{time_6_hours_ago}'
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

# Get the initial temperature for each node 6 hours ago
initial_temperatures = []
for sensor in sensor_entities:
    sensor_df = df[df['entity_id'] == sensor].sort_values(by='last_updated_ts')
    initial_temp = sensor_df.iloc[0]['state'] if not sensor_df.empty else 20  # Default to 20°C if no data
    initial_temperatures.append(initial_temp)

# Now, use the initial_temperatures array to start the simulation
n_nodes = 10  # Number of nodes
t_max = 3600* 6  # Time to simulate in seconds (e.g., 1 hour)
dt = 10  # Time step in seconds
n_steps = t_max // dt  # Number of time steps

# Heat transfer constants
k = 0.1  # Heat transfer coefficient between nodes (W/°C)
U = 12.71  # Heat loss coefficient to environment (W/°C)
c_p = 4186  # Specific heat of water (J/kg°C)
m = 50  # Mass of water in each node (kg)

# Use the real data as initial conditions
T = np.array(initial_temperatures)

# Ambient temperature
T_ambient = 20  # Ambient temperature in °C

# Store temperature over time for plotting
T_over_time = np.zeros((n_steps, n_nodes))

# Simulate over time
for t in range(n_steps):
    T_new = np.copy(T)  # Create a copy to hold new temperatures

    for i in range(n_nodes):
        # Calculate heat transfer with neighboring nodes
        if i > 0:  # Node to the left
            heat_transfer_left = k * (T[i - 1] - T[i])
        else:
            heat_transfer_left = 0

        if i < n_nodes - 1:  # Node to the right
            heat_transfer_right = k * (T[i + 1] - T[i])
        else:
            heat_transfer_right = 0

        # Calculate heat loss to the environment
        heat_loss = U * (T[i] - T_ambient)

        # Calculate the total heat flow for the node
        heat_flow = heat_transfer_left + heat_transfer_right - heat_loss

        # Update temperature using the heat flow
        T_new[i] += (heat_flow * dt) / (m * c_p)

    # Update the temperature array
    T = T_new

    # Store the temperatures for plotting
    T_over_time[t] = T

# Plot the simulated temperatures over time
time_sim = np.array([pd.to_datetime(time_6_hours_ago +  t * dt, unit='s') for t in range(n_steps)])

plt.figure(figsize=(10, 6))

# Plot the simulation data
for i in range(n_nodes):
    plt.plot(time_sim, T_over_time[:, i], label=f'sim p_{i+1}', linestyle='-', color=sensor_colors[sensor_entities[i]])

# Plot the real data for each sensor
for sensor in sensor_entities:
    sensor_df = df[df['entity_id'] == sensor]
    plt.plot(sensor_df['last_updated_ts'], sensor_df['state'], label=f'Real {sensor}', color=sensor_colors[sensor],
             linestyle='--')

# Configure the plot
plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title('Simulated vs Real Temperatures in Heat Tank')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()

# Show the plot
plt.show()
