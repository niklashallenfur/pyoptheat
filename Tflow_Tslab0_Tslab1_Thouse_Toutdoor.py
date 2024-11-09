from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import load_data

def simulate_two_layer_battery(T_source, T_battery0_measured, T_sink, params):
    # Unpack parameters
    C_source_battery0 = params['C_source_battery0']
    C_battery0_battery1 = params['C_battery0_battery1']
    C_battery1_sink = params['C_battery1_sink']

    # Initialize simulated temperatures
    T_battery0_sim = pd.Series(index=T_battery0_measured.index, dtype=float)
    T_battery1_sim = pd.Series(index=T_battery0_measured.index, dtype=float)

    # Initial conditions
    T_battery0_sim.iloc[0] = T_battery0_measured.iloc[0]
    T_battery1_sim.iloc[0] = T_battery0_measured.iloc[0]  # Assume T_battery1 starts equal to T_battery0

    # Simulate over time
    for i in range(len(T_battery0_measured.index) - 1):
        dt = (T_battery0_measured.index[i + 1] - T_battery0_measured.index[i]).total_seconds()

        # Current temperatures
        T_source_current = T_source.iloc[i]
        T_battery0_current = T_battery0_sim.iloc[i]
        T_battery1_current = T_battery1_sim.iloc[i]
        T_sink_current = T_sink.iloc[i]

        # Compute temperature changes
        dT_battery0_dt = (
                C_source_battery0 * (T_source_current - T_battery0_current) -
                C_battery0_battery1 * (T_battery0_current - T_battery1_current)
        )

        dT_battery1_dt = (
                C_battery0_battery1 * (T_battery0_current - T_battery1_current) -
                C_battery1_sink * (T_battery1_current - T_sink_current)
        )

        # Update temperatures
        T_battery0_next = T_battery0_current + dT_battery0_dt * dt
        T_battery1_next = T_battery1_current + dT_battery1_dt * dt

        # Store the next values
        T_battery0_sim.iloc[i + 1] = T_battery0_next
        T_battery1_sim.iloc[i + 1] = T_battery1_next

    # Return simulated temperatures
    simulations = pd.DataFrame({
        'T_battery0_sim': T_battery0_sim,
        'T_battery1_sim': T_battery1_sim
    })

    return simulations

from scipy.optimize import minimize, differential_evolution


def estimate_two_layer_battery_parameters(T_source, T_battery0_measured, T_sink):
    # Initial parameter guesses
    initial_params = {
        'C_source_battery0': 0.0005,
        'C_battery0_battery1': 0.005,
        'C_battery1_sink': 0.0003
    }

    # Bounds for parameters (ensure they are positive)
    bounds = [
        (1e-8, 1e-2),  # C_source_battery0
        (1e-8, 1e-2),  # C_battery0_battery1
        (1e-8, 1e-2)  # C_battery1_sink
    ]


    import multiprocessing as mp
    # Objective function to minimize
    param_names = list(initial_params.keys())
    def objective(param_values):
        # Map parameter values to names

        params = dict(zip(param_names, param_values))

        # Run simulation with current parameters
        simulations = simulate_two_layer_battery(T_source, T_battery0_measured, T_sink, params)

        # Calculate error between simulated and measured T_battery0
        error_battery0 = T_battery0_measured - simulations['T_battery0_sim']

        # Total error (sum of squared errors)
        total_error = np.sum(error_battery0 ** 2)

        print(f"Total Error: {total_error}")
        return total_error

    # Prepare initial guesses and bounds
    x0 = list(initial_params.values())

    # Perform optimization
    result = minimize(
        objective,
        x0,
        bounds=bounds,
        method='L-BFGS-B'
    )


    # Extract estimated parameters
    estimated_params = dict(zip(initial_params.keys(), result.x))

    print("Estimated Parameters:")
    for name, value in estimated_params.items():
        print(f"{name}: {value}")

    return estimated_params


## Model fitting
start_time = datetime.fromisoformat("2024-10-21T17:00:00+02:00").timestamp()
end_time = datetime.fromisoformat("2024-10-24T23:00:00+02:00").timestamp()
# start_time = datetime.fromisoformat("2024-10-21T17:00:00+02:00").timestamp()
# end_time = datetime.fromisoformat("2024-11-09T23:59:00+02:00").timestamp()

sensor_entities = ['sensor.radiator_flow_temp',
                   'sensor.radiator_return_temp',
                   'sensor.house_hall_temp',
                   'sensor.climate_coop_temperature', ]

training_data = load_data(sensor_entities, start_time, end_time)

estimated_params = estimate_two_layer_battery_parameters(
    training_data['sensor.radiator_flow_temp'],
    training_data['sensor.radiator_return_temp'],
    training_data['sensor.house_hall_temp'])

## Simulation
# Simulation interval (new interval)
sim_start_time = datetime.fromisoformat("2024-10-21T00:00:00+02:00").timestamp()
sim_end_time = datetime.fromisoformat("2024-11-09T23:59:00+02:00").timestamp()
simulation_data = load_data(sensor_entities, sim_start_time, sim_end_time)

T_flow = simulation_data['sensor.radiator_flow_temp']
T_return = simulation_data['sensor.radiator_return_temp']
T_house = simulation_data['sensor.house_hall_temp']
T_outdoor = simulation_data['sensor.climate_coop_temperature']

simulations = simulate_two_layer_battery(T_flow, T_return, T_house, estimated_params)




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
for sim_label in simulations.columns:
    axSim.plot(simulations.index, simulations[sim_label], label=sim_label, linestyle='--')

format_temp_axis(axSim)

plt.tight_layout()
plt.show()

# radiator_temp_diff = (radiator_flow_temp - radiator_return_temp) #.apply(lambda x: x if 0 < x else 0)
# radiator_power = radiator_temp_diff * 980
