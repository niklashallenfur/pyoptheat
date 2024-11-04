import pulp as pl

# Constants
num_hours = 24
heat_demand_l1 = 100  # W for l1
heat_demand_l2 = 1000  # W for l2
min_temp_l1 = 40  # Minimum required temperature for l1
min_temp_l2 = 25  # Minimum required temperature for l2
max_power = 5000  # Maximum power the heater can use
efficiency = 0.95  # Efficiency of heater
electricity_prices = [0.1] * num_hours  # Example prices per hour (replace with real data)

# Initialize model
model = pl.LpProblem("Heating_Cost_Optimization", pl.LpMinimize)

# Variables
T_l1 = [pl.LpVariable(f"T_l1_{t}", lowBound=min_temp_l1) for t in range(num_hours + 1)]
T_l2 = [pl.LpVariable(f"T_l2_{t}", lowBound=min_temp_l2) for t in range(num_hours + 1)]
P_el = [pl.LpVariable(f"P_el_{t}", lowBound=0, upBound=max_power) for t in range(num_hours)]

# Auxiliary variable for mixing between l1 and l2
r_mix = [pl.LpVariable(f"r_mix_{t}", lowBound=0, upBound=1) for t in range(num_hours)]

# Objective: Minimize total electricity cost over 24 hours
model += pl.lpSum([P_el[t] * electricity_prices[t] for t in range(num_hours)])

# Constraints
for t in range(num_hours):
    # Total power for heating must not exceed the maximum power
    model += P_el[t] <= max_power

    # Power allocation between l1 and l2, based on the mixing ratio r_mix
    model += r_mix[t] * P_el[t] <= heat_demand_l1  # Power going to l1
    model += (1 - r_mix[t]) * P_el[t] <= heat_demand_l2  # Power going to l2

    # Energy balance for l1 and l2, with efficiency considerations
    model += T_l1[t + 1] == T_l1[t] + r_mix[t] * P_el[t] * efficiency / (heat_demand_l1 + heat_demand_l2)
    model += T_l2[t + 1] == T_l2[t] + (1 - r_mix[t]) * P_el[t] * efficiency / (heat_demand_l1 + heat_demand_l2)

# Solve the model
model.solve()

# Output results
if pl.LpStatus[model.status] == "Optimal":
    print("Optimal solution found!")
    for t in range(num_hours):
        print(f"Hour {t + 1}:")
        print(f"  Power: {P_el[t].value()} W")
        print(f"  Temp L1: {T_l1[t].value()} °C")
        print(f"  Temp L2: {T_l2[t].value()} °C")
        print(f"  Mixing Ratio (r_mix): {r_mix[t].value()}")
else:
    print("No optimal solution found.")
