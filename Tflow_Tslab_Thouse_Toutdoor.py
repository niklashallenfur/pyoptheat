from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from load_data import load_data


def fit_source_battery_sink(T_source, T_battery, T_sink):
    # Calculate the rate of change of T_slab (radiator_return_temp)
    delta_t = 60  # Time interval in seconds (since data is at 1-minute intervals)

    # Compute dT_slab/dt in °C per second
    dT_battery_dt = ((T_battery.diff() / delta_t)
                     .dropna())

    # Ensure all series are aligned with dT_battery_dt
    T_source = T_source.loc[dT_battery_dt.index]
    T_battery = T_battery.loc[dT_battery_dt.index]
    T_sink = T_sink.loc[dT_battery_dt.index]

    # Compute X1 and X2
    Heating = T_source - T_battery
    Consuming = T_sink - T_battery

    from sklearn.linear_model import LinearRegression

    # Prepare the data for regression
    # Remove any potential NaN values due to differencing
    regression_data = pd.DataFrame({
        'dT_dt': dT_battery_dt,
        'Heating': Heating,
        'Consuming': Consuming
    }).dropna()

    # Variables for regression
    Y = regression_data['dT_dt'].values  # Dependent variable
    X = regression_data[['Heating', 'Consuming']].values  # Independent variables

    # Perform linear regression without intercept
    model = LinearRegression(fit_intercept=False)
    model.fit(X, Y)

    # Extract coefficients
    C_source = model.coef_[0]
    C_sink = model.coef_[1]

    print(f"Estimated C_source: {C_source}")
    print(f"Estimated C_sink: {C_sink}")

    # Calculate R^2 score
    r_squared = model.score(X, Y)
    print(f"R-squared: {r_squared}")
    return C_source, C_sink


def simulate_source_battery_sink(T_source, T_battery, T_sink, C_source, C_sink):
    # Define simulation start times (every 3 hours within the data range)
    simulation_start_times = pd.date_range(
        start=T_battery.index[0],
        end=T_battery.index[-1],
        freq='12h'
    )
    simulations = pd.DataFrame(index=T_battery.index)
    # Loop over each simulation start time
    for start_time in simulation_start_times:
        if start_time not in T_battery.index:
            continue  # Skip if start time isn't in the index

        # Find the starting index
        start_idx = T_battery.index.get_loc(start_time)

        # Initialize the simulated T_slab series
        T_battery_sim = pd.Series(index=T_battery.index, dtype=float)
        T_battery_sim.iloc[:start_idx] = np.nan  # Before simulation starts
        T_battery_sim.iloc[start_idx] = T_battery.iloc[start_idx]  # Initial condition

        # Simulate from start_idx onwards
        for i in range(start_idx, len(T_battery.index) - 1):
            dt = (T_battery.index[i + 1] - T_battery.index[i]).total_seconds()

            # Current temperatures
            T_battery_current = T_battery_sim.iloc[i]
            T_source_current = T_source.iloc[i]
            T_sink_current = T_sink.iloc[i]

            # Calculate the rate of change
            dT_battery_dt_current = (
                    C_source * (T_source_current - T_battery_current) +
                    C_sink * (T_sink_current - T_battery_current)
            )

            # Update T_slab using Euler's method
            T_battery_next = T_battery_current + dT_battery_dt_current * dt

            # Store the next value
            T_battery_sim.iloc[i + 1] = T_battery_next

        # Add the simulation to the DataFrame
        sim_label = f'Simulation_{start_time.strftime("%Y-%m-%d %H:%M")}'
        simulations[sim_label] = T_battery_sim
        break
    return simulations


## Model fitting
# start_time = datetime.fromisoformat("2024-10-21T17:00:00+02:00").timestamp()
# end_time = datetime.fromisoformat("2024-10-24T23:00:00+02:00").timestamp()
start_time = datetime.fromisoformat("2024-10-21T17:00:00+02:00").timestamp()
end_time = datetime.fromisoformat("2024-11-09T23:59:00+02:00").timestamp()

sensor_entities = ['sensor.radiator_flow_temp',
                   'sensor.radiator_return_temp',
                   'sensor.house_hall_temp',
                   'sensor.climate_coop_temperature', ]

training_data = load_data(sensor_entities, start_time, end_time)

C_flow_radiator, C_radiator_house = fit_source_battery_sink(training_data['sensor.radiator_flow_temp'],
                                                            training_data['sensor.radiator_return_temp'],
                                                            training_data['sensor.house_hall_temp'])

C_radiator_house2, C_house_outdoor = fit_source_battery_sink(training_data['sensor.radiator_return_temp'],
                                                             training_data['sensor.house_hall_temp'],
                                                             training_data['sensor.climate_coop_temperature'])

## Simulation
# Simulation interval (new interval)
sim_start_time = datetime.fromisoformat("2024-10-21T00:00:00+02:00").timestamp()
sim_end_time = datetime.fromisoformat("2024-11-09T23:59:00+02:00").timestamp()
simulation_data = load_data(sensor_entities, sim_start_time, sim_end_time)

T_flow = simulation_data['sensor.radiator_flow_temp']
T_return = simulation_data['sensor.radiator_return_temp']
T_house = simulation_data['sensor.house_hall_temp']
T_outdoor = simulation_data['sensor.climate_coop_temperature']

radiator_simulations = simulate_source_battery_sink(T_flow, T_return, T_house, C_flow_radiator, C_radiator_house)
house_simulations = simulate_source_battery_sink(T_return, T_house, T_outdoor, C_radiator_house2, C_house_outdoor)


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
f, axSim = plt.subplots(1, 1, figsize=(40, 10), sharex=True)
axSim.plot(T_flow.index, T_flow, label="T_flow", color="#FFAAAA")
axSim.plot(T_return.index, T_return, label="T_slab", color="#AAAAFF")
axSim.plot(T_house.index, T_house, label="T_house", color="#AAFFAA")
axSim.plot(T_outdoor.index, T_outdoor, label="T_outdoor", color="#0055AA")
for sim_label in radiator_simulations.columns:
    axSim.plot(radiator_simulations.index, radiator_simulations[sim_label], label="radiator_sim", linestyle='--')
for sim_label in house_simulations.columns:
    axSim.plot(house_simulations.index, house_simulations[sim_label], label="house_sim", linestyle='--')

format_temp_axis(axSim)

plt.tight_layout()
plt.show()

# radiator_temp_diff = (radiator_flow_temp - radiator_return_temp) #.apply(lambda x: x if 0 < x else 0)
# radiator_power = radiator_temp_diff * 980