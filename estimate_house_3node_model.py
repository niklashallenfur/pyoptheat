import json
from datetime import datetime
from pathlib import Path

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from load_data import load_data


def compute_capacities_and_resistances():
    """Return fixed C_a, C_s, C_b and R network based on user assumptions.

    - Slab: concrete 11 x 9 x 0.4 m, rho ~ 2400 kg/m³, cp ~ 880 J/(kg*K)
    - Air volume: 135 m² * 2.5 m, rho ~ 1.2 kg/m³, cp ~ 1000 J/(kg*K)
    - Total R from slab to outdoor ~ 6.7 K/kW => 0.0067 K/W

    We split the overall R into three parts for the 3-node chain:
        slab --R_sa-- air --R_ab-- structure --R_bo-- outdoor
    such that R_sa + R_ab + R_bo = R_total.
    """
    # Slab capacity
    slab_area = 11.0 * 9.0  # m²
    # slab_thickness = 0.42    # m
    slab_thickness = 0.06    # m
    slab_volume = slab_area * slab_thickness
    rho_concrete = 2400.0    # kg/m³
    cp_concrete = 880.0      # J/(kg*K)
    slab_mass = slab_volume * rho_concrete
    C_s = slab_mass * cp_concrete

    # Air capacity (single effective zone)
    floor_area = 340.0       # m²
    height = 2.5             # m
    air_volume = floor_area * height
    rho_air = 1.2            # kg/m³
    cp_air = 1000.0          # J/(kg*K)
    air_mass = air_volume * rho_air
    C_a = air_mass * cp_air

    # Structure capacity: less certain, assume, say, 5x air capacity as a starting point
    C_b = 10.0 * 3600 * 1000  # J/K (equiv. to 5x air capacity)

    # Total R from slab to outdoor (K/W)
    R_total = 8.3 / 1000.0  # K/kW => K/W

    # Split R_total into three segments; air-structure typically large, so bias there
    R_sa = 0.1 * R_total
    R_ab = 0.4 * R_total
    R_bo = 0.5 * R_total

    return C_a, C_s, C_b, R_sa, R_ab, R_bo


def steady_state_initial_conditions(Ta0, P_rad0, Tout0, C_a, C_s, C_b, R_sa, R_ab, R_bo):
    """Compute steady-state Ts0, Tb0 for given Ta0, P_rad0, Tout0.

    dTs/dt = 0: 0 = P_rad0 + (Ta0 - Ts0)/R_sa
      => Ts0 = Ta0 + P_rad0 * R_sa

    dTb/dt = 0: 0 = (Ta0 - Tb0)/R_ab + (Tout0 - Tb0)/R_bo
      => Tb0 = (Ta0/R_ab + Tout0/R_bo) / (1/R_ab + 1/R_bo)

    C's are not needed directly here, but included for completeness/signature symmetry.
    """
    Ts0 = Ta0 + P_rad0 * R_sa
    Tb0 = (Ta0 / R_ab + Tout0 / R_bo) / (1.0 / R_ab + 1.0 / R_bo)
    return Ts0, Tb0


def simulate_3node(idx, Tout, P_rad, Ta_obs,
                    C_a, C_s, C_b, R_sa, R_ab, R_bo):
    """Simulate the 3-node model forward in time on a given index.

    States: Ts (slab), Ta (air), Tb (structure)

    C_s dTs/dt = P_rad + (Ta - Ts)/R_sa
    C_a dTa/dt = (Ts - Ta)/R_sa + (Tb - Ta)/R_ab
    C_b dTb/dt = (Ta - Tb)/R_ab + (Tout - Tb)/R_bo
    """
    idx = pd.to_datetime(idx)
    dt_sec = idx.to_series().diff().dt.total_seconds().fillna(0.0).astype(float).to_numpy()

    Tout = np.asarray(Tout, dtype=float)
    P_rad = np.asarray(P_rad, dtype=float)
    Ta_obs = np.asarray(Ta_obs, dtype=float)

    n = len(idx)
    Ts = np.empty(n, dtype=float)
    Ta = np.empty(n, dtype=float)
    Tb = np.empty(n, dtype=float)

    # Initial conditions from first sample steady-state assumption
    Ta[0] = Ta_obs[0]
    Ts[0], Tb[0] = steady_state_initial_conditions(
        Ta[0], P_rad[0], Tout[0], C_a, C_s, C_b, R_sa, R_ab, R_bo
    )

    for k in range(1, n):
        dt = dt_sec[k]
        if dt <= 0:
            Ts[k] = Ts[k-1]
            Ta[k] = Ta[k-1]
            Tb[k] = Tb[k-1]
            continue

        # Use previous state and inputs as explicit Euler
        Ts_prev = Ts[k-1]
        Ta_prev = Ta[k-1]
        Tb_prev = Tb[k-1]
        Tout_prev = Tout[k-1]
        P_rad_prev = P_rad[k-1]

        dTsdt = (P_rad_prev + (Ta_prev - Ts_prev) / R_sa) / C_s
        dTadt = ((Ts_prev - Ta_prev) / R_sa + (Tb_prev - Ta_prev) / R_ab) / C_a
        dTbdt = ((Ta_prev - Tb_prev) / R_ab + (Tout_prev - Tb_prev) / R_bo) / C_b

        Ts[k] = Ts_prev + dTsdt * dt
        Ta[k] = Ta_prev + dTadt * dt
        Tb[k] = Tb_prev + dTbdt * dt

    return Ts, Ta, Tb


def main():
    # Sensors (same as in estimate_house_heat_exchange.py)
    tout = 'sensor.torild_air_temperature'
    thouse = 'sensor.house_hall_temp'
    pradiator = 'sensor.radiator_power'
    tflow = 'sensor.radiator_flow_temp'
    treturn = 'sensor.radiator_return_temp'

    # Constant internal gains (W) from occupants/plug loads, added to radiator power
    constant_appliances_power = 600.0

    stockholm_tz = ZoneInfo('Europe/Stockholm')

    # Use same long period for now
    start_stockholm = datetime(2025, 1, 15, 0, 0, 0).replace(tzinfo=stockholm_tz)
    end_stockholm = datetime(2025, 2, 16, 0, 0, 0).replace(tzinfo=stockholm_tz)
    
    # start_stockholm = datetime(2025, 11, 1, 0, 0, 0).replace(tzinfo=stockholm_tz)
    # end_stockholm = datetime(2025, 11, 17, 0, 0, 0).replace(tzinfo=stockholm_tz)

    start_utc = start_stockholm.astimezone(ZoneInfo('UTC'))
    end_utc = end_stockholm.astimezone(ZoneInfo('UTC'))

    print(f"Loading house data from {start_stockholm.isoformat()} to {end_stockholm.isoformat()} (Stockholm time)")
    print(f"UTC times: {start_utc.isoformat()} to {end_utc.isoformat()}")

    df = load_data([tout, thouse, pradiator, tflow, treturn], from_time=start_utc.timestamp(), to_time=end_utc.timestamp())
    if df.empty:
        raise ValueError("No data found for the specified period")

    print(f"Loaded {len(df)} samples from {df.index[0]} to {df.index[-1]}")

    radiator_w = df[pradiator].astype(float).fillna(0.0).clip(lower=0.0)
    df['P_total'] = radiator_w + constant_appliances_power

    # smooth power for plotting and fitting: 60-minute average to reduce noise
    try:
        df['P_total_smooth'] = df['P_total'].rolling('60min', min_periods=1).mean()
    except Exception:
        df['P_total_smooth'] = df['P_total'].rolling(60, min_periods=1).mean()

    # Get fixed parameters
    C_a, C_s, C_b, R_sa, R_ab, R_bo = compute_capacities_and_resistances()
    print("Capacities/Resistances:")
    print(f"  C_a (air)   = {C_a/3_600_000:.3f} kWh/K")
    print(f"  C_s (slab)  = {C_s/3_600_000:.3f} kWh/K")
    print(f"  C_b (struct)= {C_b/3_600_000:.3f} kWh/K")
    print(f"  C_total     = {(C_a + C_s + C_b)/3_600_000:.3f} kWh/K")
    print(f"  R_sa        = {R_sa*1000:.3f} K/kW")
    print(f"  R_ab        = {R_ab*1000:.3f} K/kW")
    print(f"  R_bo        = {R_bo*1000:.3f} K/kW")
    R_total = R_sa + R_ab + R_bo
    print(f"  R_total     = {R_total:.3e} K/W  ({R_total*1000:.2f} K/kW)")

    idx = df.index
    # Use 60-minute averaged temperatures to reduce noise
    Tout_vals = df[tout].astype(float).rolling('60min', min_periods=1).mean().to_numpy()
    Ta_obs_vals = df[thouse].astype(float).rolling('60min', min_periods=1).mean().to_numpy()

    # 60-minute averaged radiator flow and return temps for comparison with simulated slab temp
    Tflow_60 = df[tflow].astype(float).rolling('60min', min_periods=1).mean().to_numpy()
    Treturn_60 = df[treturn].astype(float).rolling('60min', min_periods=1).mean().to_numpy()
    # Use smoothed power for simulation/fitting to reduce noise
    P_total_vals = df['P_total_smooth'].astype(float).to_numpy()
    
    # save values to 

    Ts_sim, Ta_sim, Tb_sim = simulate_3node(
        idx,
        Tout_vals,
        P_total_vals,
        Ta_obs_vals,
        C_a=C_a,
        C_s=C_s,
        C_b=C_b,
        R_sa=R_sa,
        R_ab=R_ab,
        R_bo=R_bo,
    )

    df['T_air_sim'] = Ta_sim
    df['T_slab_sim'] = Ts_sim
    df['T_struct_sim'] = Tb_sim

    # Simple error metric
    mse = float(np.mean((Ta_sim - Ta_obs_vals) ** 2))
    rmse = float(np.sqrt(mse))
    print(f"RMSE between simulated and observed indoor temp: {rmse:.3f} °C")

    # Save parameters & basic stats
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    period_tag = f"{start_stockholm.date()}_to_{end_stockholm.date()}"
    out_json = output_dir / f"house_3node_{period_tag}.json"

    output_data = {
        'period': {
            'start_stockholm': start_stockholm.isoformat(),
            'end_stockholm': end_stockholm.isoformat(),
            'start_utc': start_utc.isoformat(),
            'end_utc': end_utc.isoformat(),
        },
        'sensors': {
            'Tout': tout,
            'Thouse': thouse,
            'Pradiator': pradiator,
        },
        'constant_appliances_power_W': constant_appliances_power,
        'parameters': {
            'C_a_J_per_K': C_a,
            'C_s_J_per_K': C_s,
            'C_b_J_per_K': C_b,
            'R_sa_K_per_W': R_sa,
            'R_ab_K_per_W': R_ab,
            'R_bo_K_per_W': R_bo,
        },
        'metrics': {
            'RMSE_Thouse_degC': rmse,
        },
    }

    with out_json.open('w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved parameters and metrics to {out_json}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))

    ax.plot(idx, Tout_vals, label='Outdoor temp (observed)', color='#1f77b4', linewidth=1.5)
    ax.plot(idx, Ta_obs_vals, label='House temp (observed)', color='#ff7f0e', linewidth=2)
    ax.plot(idx, Ta_sim, label='House temp (simulated, 3-node)', color='#2ca02c', linestyle='--', linewidth=2)
    ax.plot(idx, Ts_sim, label='Slab temp (simulated)', color='#9467bd', linestyle=':', linewidth=1.5)
    ax.plot(idx, Tb_sim, label='Structure temp (simulated)', color='#8c564b', linestyle='-.', linewidth=1.5)

    # Radiator flow/return temps (60-min mean) to visually compare with slab temp
    ax.plot(idx, Tflow_60, label='Radiator flow temp (60-min mean)', color='#d62728', linewidth=1.5, alpha=0.8)
    ax.plot(idx, Treturn_60, label='Radiator return temp (60-min mean)', color='#17becf', linewidth=1.5, alpha=0.8)

    ax2 = ax.twinx()
    ax2.plot(idx, df['P_total_smooth'], label='Total power (W, 5-min mean)', color='#6a51a3', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Power (W)', fontsize=14)
    ax2.tick_params(axis='y', labelsize=12)

    ax.set_title(
        f"3-node house model: RMSE = {rmse:.2f} °C, R_total = {(R_sa + R_ab + R_bo)*1000:.2f} K/kW",
        fontsize=18,
    )
    ax.set_xlabel('Time (Stockholm timezone)', fontsize=16)
    ax.set_ylabel('Temperature (°C)', fontsize=16)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))

    ax.tick_params(axis='x', rotation=45, labelsize=12, which='major')
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(True, alpha=0.3)

    # Merge legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=12)

    plt.tight_layout()

    png_path = output_dir / f"house_3node_{period_tag}.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot to {png_path}")

    try:
        plt.show()
    except Exception:
        plt.close(fig)


if __name__ == '__main__':
    main()
