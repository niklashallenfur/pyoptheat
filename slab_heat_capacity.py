from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from fit_source_battery_sink import fit_source_battery_sink
from load_data import load_data
from simulate_slab_heat import simulate_source_battery_sink

start_time = datetime.fromisoformat("2024-10-21T17:00:00+02:00").timestamp()
# end_time = datetime.fromisoformat("2024-10-24T23:00:00+02:00").timestamp()
# start_time = datetime.fromisoformat("2024-10-19T17:00:00+02:00").timestamp()
end_time = datetime.fromisoformat("2024-11-01T23:59:00+02:00").timestamp()

sensor_entities = ['sensor.radiator_flow_temp',
                   'sensor.radiator_return_temp',
                   'sensor.house_hall_temp']

training_data = load_data(sensor_entities, start_time, end_time)

## Model fitting
C_source, C_sink = fit_source_battery_sink(training_data['sensor.radiator_flow_temp'],
                                           training_data['sensor.radiator_return_temp'],
                                           training_data['sensor.house_hall_temp'])

# Simulation interval (new interval)
sim_start_time = datetime.fromisoformat("2024-10-27T16:00:00+02:00").timestamp()
sim_end_time = datetime.fromisoformat("2024-11-01T23:59:00+02:00").timestamp()
simulation_data = load_data(sensor_entities, sim_start_time, sim_end_time)

T_flow_sim = simulation_data['sensor.radiator_flow_temp']
T_return_sim = simulation_data['sensor.radiator_return_temp']
T_house_sim = simulation_data['sensor.house_hall_temp']

simulations = simulate_source_battery_sink(T_flow_sim, T_return_sim, T_house_sim, C_source, C_sink)


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
    axSim.plot(T_flow_sim.index, T_flow_sim, label="T_flow", color="#FFAAAA")
    axSim.plot(T_return_sim.index, T_return_sim, label="T_slab", color="#AAAAFF")
    axSim.plot(T_house_sim.index, T_house_sim, label="T_house", color="#AAFFAA")
    axSim.plot(simulations.index, simulations[sim_label], label=sim_label, linestyle='--')
    format_temp_axis(axSim)

plt.tight_layout()
plt.show()

# radiator_temp_diff = (radiator_flow_temp - radiator_return_temp) #.apply(lambda x: x if 0 < x else 0)
# radiator_power = radiator_temp_diff * 980
