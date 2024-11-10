from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import load_data

# Import skopt modules
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

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

# Define the search space for the parameters
from skopt.space import Real

# Define parameter ranges
space  = [
    Real(1e-8, 1e-2, name='C_source_battery0', prior='log-uniform'),
    Real(1e-8, 1e-2, name='C_battery0_battery1', prior='log-uniform'),
    Real(1e-8, 1e-2, name='C_battery1_sink', prior='log-uniform')
]

# Define the objective function
def objective_skopt(params_list):
    # Map the list of parameters to their names
    params = dict(zip(['C_source_battery0', 'C_battery0_battery1', 'C_battery1_sink'], params_list))

    # Run simulation with current parameters
    simulations = simulate_two_layer_battery(T_source, T_battery0_measured, T_sink, params)

    # Calculate error between simulated and measured T_battery0
    error_battery0 = T_battery0_measured - simulations['T_battery0_sim']

    # Total error (sum of squared errors)
    total_error = np.sum(error_battery0 ** 2)

    return total_error

# Use the decorator to pass parameters by name
@use_named_args(space)
def skopt_objective_wrapper(**params):
    # Run simulation with current parameters
    simulations = simulate_two_layer_battery(T_source, T_battery0_measured, T_sink, params)

    # Calculate error between simulated and measured T_battery0
    error_battery0 = T_battery0_measured - simulations['T_battery0_sim']

    # Total error (sum of squared errors)
    total_error = np.sum(error_battery0 ** 2)

    return total_error

def estimate_two_layer_battery_parameters_skopt(T_source_input, T_battery0_measured_input, T_sink_input):
    global T_source, T_battery0_measured, T_sink
    # Assign the inputs to global variables used in the objective function
    T_source = T_source_input
    T_battery0_measured = T_battery0_measured_input
    T_sink = T_sink_input

    # Run the optimization
    result = gp_minimize(
        func=skopt_objective_wrapper,
        dimensions=space,
        n_calls=150,
        n_random_starts=15,
        random_state=42,
        n_jobs=-1  # Use all available CPUs
    )

    # Extract estimated parameters
    estimated_params = dict(zip(['C_source_battery0', 'C_battery0_battery1', 'C_battery1_sink'], result.x))

    print("Estimated Parameters:")
    for name, value in estimated_params.items():
        print(f"{name}: {value}")

    return estimated_params

## Model fitting
start_time = datetime.fromisoformat("2024-10-21T17:00:00+02:00").timestamp()
end_time = datetime.fromisoformat("2024-11-09T23:00:00+02:00").timestamp()

sensor_entities = [
    'sensor.radiator_flow_temp',
    'sensor.radiator_return_temp',
    'sensor.house_hall_temp',
    'sensor.climate_coop_temperature',
]

training_data = load_data(sensor_entities, start_time, end_time)

# Extract temperature series
T_flow = training_data['sensor.radiator_flow_temp']
T_return = training_data['sensor.radiator_return_temp']
T_house = training_data['sensor.house_hall_temp']
T_outdoor = training_data['sensor.climate_coop_temperature']

# Estimate parameters using skopt
estimated_params = estimate_two_layer_battery_parameters_skopt(
    T_source_input=T_flow,
    T_battery0_measured_input=T_return,
    T_sink_input=T_house
)

## Simulation
# Simulation interval (new interval)
sim_start_time = datetime.fromisoformat("2024-10-21T00:00:00+02:00").timestamp()
sim_end_time = datetime.fromisoformat("2024-11-09T23:59:00+02:00").timestamp()
simulation_data = load_data(sensor_entities, sim_start_time, sim_end_time)

T_flow_sim = simulation_data['sensor.radiator_flow_temp']
T_return_sim = simulation_data['sensor.radiator_return_temp']
T_house_sim = simulation_data['sensor.house_hall_temp']
T_outdoor_sim = simulation_data['sensor.climate_coop_temperature']

simulations = simulate_two_layer_battery(T_flow_sim, T_return_sim, T_house_sim, estimated_params)

# Visualize values in plots

def format_temp_axis(axs):
    axs.set_xlabel('Time')
    axs.set_ylabel('Temperature (°C)')
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M:%S'))
    axs.xaxis.set_major_locator(mdates.AutoDateLocator())
    axs.tick_params(axis='x', rotation=45)
    axs.grid(True)
    axs.legend(loc='best')

# Plot the simulations
f, axSim = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
axSim.plot(T_flow_sim.index, T_flow_sim, label="T_flow", color="#FFAAAA")
axSim.plot(T_return_sim.index, T_return_sim, label="T_battery0 (Measured)", color="#AAAAFF")
axSim.plot(T_house_sim.index, T_house_sim, label="T_house", color="#AAFFAA")
axSim.plot(T_outdoor_sim.index, T_outdoor_sim, label="T_outdoor", color="#0055AA")

axSim.plot(simulations.index, simulations['T_battery0_sim'], label="T_battery0 (Simulated)", linestyle='--')
axSim.plot(simulations.index, simulations['T_battery1_sim'], label="T_battery1 (Simulated)", linestyle='--')

format_temp_axis(axSim)

plt.tight_layout()
plt.show()
