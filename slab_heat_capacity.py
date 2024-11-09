from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from load_data import load_data

start_time = datetime.fromisoformat("2024-10-21T17:00:00+02:00").timestamp()
end_time = datetime.fromisoformat("2024-10-24T23:00:00+02:00").timestamp()
# start_time = datetime.fromisoformat("2024-10-19T17:00:00+02:00").timestamp()
# end_time = datetime.fromisoformat("2024-10-25T23:00:00+02:00").timestamp()

sensor_entities = ['sensor.radiator_flow_temp',
                   'sensor.radiator_return_temp',
                   'sensor.house_hall_temp']

training_data = load_data(sensor_entities, start_time, end_time)

radiator_flow_temp = training_data['sensor.radiator_flow_temp']
radiator_return_temp = training_data['sensor.radiator_return_temp']
house_hall_temp = training_data['sensor.house_hall_temp']

radiator_temp_diff = (radiator_flow_temp - radiator_return_temp).apply(lambda x: x if 0 < x else 0)
radiator_power = radiator_temp_diff * 980

## Model fitting

# Calculate the rate of change of T_slab (radiator_return_temp)
delta_t = 60  # Time interval in seconds (since data is at 1-minute intervals)

# Compute dT_slab/dt in °C per second
dT_slab_dt = (radiator_return_temp.diff()) / delta_t

# Remove NaN values introduced by diff()
dT_slab_dt = dT_slab_dt.dropna()

# Ensure all series are aligned with dT_slab_dt
T_flow = radiator_flow_temp.loc[dT_slab_dt.index]
T_return = radiator_return_temp.loc[dT_slab_dt.index]
T_house = house_hall_temp.loc[dT_slab_dt.index]

# Compute X1 and X2
X1 = T_flow - T_return
X2 = T_house - T_return

import numpy as np
from sklearn.linear_model import LinearRegression

# Prepare the data for regression
# Remove any potential NaN values due to differencing
regression_data = pd.DataFrame({
    'dT_slab_dt': dT_slab_dt,
    'X1': X1,
    'X2': X2
}).dropna()

# Variables for regression
Y = regression_data['dT_slab_dt'].values  # Dependent variable
X = regression_data[['X1', 'X2']].values  # Independent variables

# Perform linear regression without intercept
model = LinearRegression(fit_intercept=False)
model.fit(X, Y)

# Extract coefficients
C_heat = model.coef_[0]
C_house = model.coef_[1]

print(f"Estimated C_heat: {C_heat}")
print(f"Estimated C_house: {C_house}")

# Calculate R^2 score
r_squared = model.score(X, Y)
print(f"R-squared: {r_squared}")

# Simulate values
# Predicted dT_slab/dt
Y_pred = model.predict(X)


def simulate(T_slab, T_flow, T_house, C_heat, C_house):
    # Define simulation start times (every 3 hours within the data range)
    simulation_start_times = pd.date_range(
        start=T_slab.index[0],
        end=T_slab.index[-1],
        freq='6H'
    )
    simulations = pd.DataFrame(index=T_slab.index)
    # Loop over each simulation start time
    for start_time in simulation_start_times:
        if start_time not in T_slab.index:
            continue  # Skip if start time isn't in the index

        # Find the starting index
        start_idx = T_slab.index.get_loc(start_time)

        # Initialize the simulated T_slab series
        T_slab_sim = pd.Series(index=T_slab.index, dtype=float)
        T_slab_sim.iloc[:start_idx] = np.nan  # Before simulation starts
        T_slab_sim.iloc[start_idx] = T_slab.iloc[start_idx]  # Initial condition

        # Simulate from start_idx onwards
        for i in range(start_idx, len(T_slab.index) - 1):
            dt = (T_slab.index[i + 1] - T_slab.index[i]).total_seconds()

            # Current temperatures
            T_slab_current = T_slab_sim.iloc[i]
            T_flow_current = T_flow.iloc[i]
            T_house_current = T_house.iloc[i]

            # Calculate the rate of change
            dT_slab_dt_current = (
                    C_heat * (T_flow_current - T_slab_current) +
                    C_house * (T_house_current - T_slab_current)
            )

            # Update T_slab using Euler's method
            T_slab_next = T_slab_current + dT_slab_dt_current * dt

            # Store the next value
            T_slab_sim.iloc[i + 1] = T_slab_next

        # Add the simulation to the DataFrame
        sim_label = f'Simulation_{start_time.strftime("%Y-%m-%d %H:%M")}'
        simulations[sim_label] = T_slab_sim
    return simulations


simulations = simulate(T_return, T_flow, T_house, C_heat, C_house)


# Visualize values in plots

def format_temp_axis(axs):
    global lines_1, labels_1
    axs.set_xlabel('Time')
    axs.set_ylabel('Temperature (°C)')
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M:%S'))
    axs.xaxis.set_major_locator(mdates.AutoDateLocator())
    axs.tick_params(axis='x', rotation=45)
    axs.grid(True)
    lines_1, labels_1 = axs.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    axs.legend(lines_1
               # + lines_2,
               , labels_1
               # + labels_2
               , loc='best')


# Plot the simulations
for sim_label in simulations.columns:
    f, axSim = plt.subplots(1, 1, figsize=(40, 15), sharex=True)
    axSim.plot(T_flow.index, T_flow, label="T_flow", color="#FFAAAA")
    axSim.plot(T_return.index, T_return, label="T_slab", color="#AAAAFF")
    axSim.plot(T_house.index, T_house, label="T_house", color="#AAFFAA")
    axSim.plot(simulations.index, simulations[sim_label], label=sim_label, linestyle='--')
    format_temp_axis(axSim)

plt.tight_layout()
plt.show()
