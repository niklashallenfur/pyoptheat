import numpy as np
import pandas as pd


def simulate_source_battery_sink(T_source, T_battery, T_sink, C_source, C_sink):
    # Define simulation start times (every 3 hours within the data range)
    simulation_start_times = pd.date_range(
        start=T_battery.index[0],
        end=T_battery.index[-1],
        freq='12H'
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
