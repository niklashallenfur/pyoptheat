import json
from datetime import datetime
from pathlib import Path

from typing import Dict

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from load_data import load_data
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar


def fit_tau_from_cooling(
    df: pd.DataFrame,
    tout_col: str,
    thouse_col: str,
    *,
    k_bounds: tuple[float, float] = (1e-7, 1e-3),
    max_gap_seconds: float | None = None,
) -> Dict[str, float]:
    """
    Time-domain fit of the thermal time constant tau over a cooling interval.

    Model (piecewise-constant inputs over each sample interval Δt_i):
        T_{i+1} = a_i T_i + (1 - a_i) * (Tout_i + b/k),
        a_i = exp(-k * Δt_i),
    where k = 1/tau [1/s] and b [°C/s] is a constant bias capturing internal gains.

    We fit k by minimizing the squared residuals of the exact update equation,
    and for each candidate k we solve the optimal b in closed form.
    """
    df = df.copy()

    # Build vectors and time steps
    t_series = df.index.to_series()
    dt_sec = t_series.diff().dt.total_seconds().astype(float).to_numpy()
    T = df[thouse_col].astype(float).to_numpy()
    Tout = df[tout_col].astype(float).to_numpy()

    # Drop first sample (dt=nan) and any nonpositive dt
    mask = np.isfinite(dt_sec) & (dt_sec > 0)
    mask[0] = False  # ensure first is dropped
    idx = np.where(mask)[0]
    if len(idx) < 10:
        raise ValueError("Not enough valid sample intervals to fit tau")

    dt = dt_sec[idx]
    T_i = T[idx - 1]
    T_ip1 = T[idx]
    Tout_i = Tout[idx - 1]

    if max_gap_seconds is not None:
        gap_mask = dt <= float(max_gap_seconds)
        dt = dt[gap_mask]
        T_i = T_i[gap_mask]
        T_ip1 = T_ip1[gap_mask]
        Tout_i = Tout_i[gap_mask]

    def sse_for_k(k: float) -> tuple[float, float]:
        if not np.isfinite(k) or k <= 0.0:
            return (np.inf, 0.0)
        a = np.exp(-k * dt)
        one_minus_a = (1.0 - a)
        # y = T_{i+1} - a*T_i - (1-a)*Tout_i = (1-a)*(b/k)
        y = T_ip1 - a * T_i - one_minus_a * Tout_i
        denom = np.dot(one_minus_a, one_minus_a)
        if denom <= 0:
            return (np.inf, 0.0)
        c_hat = float(np.dot(one_minus_a, y) / denom)  # c = b/k
        b_hat = k * c_hat
        residuals = y - one_minus_a * c_hat
        sse = float(np.dot(residuals, residuals))
        return (sse, b_hat)

    # 1D bounded search for k
    bounds = (float(k_bounds[0]), float(k_bounds[1]))
    res = minimize_scalar(lambda kk: sse_for_k(kk)[0], bounds=bounds, method='bounded', options={'xatol': 1e-12})
    k_opt = float(res.x)
    sse_opt, b_opt = sse_for_k(k_opt)

    # Simulate full series to compute R² on temperatures
    a_full = np.exp(-k_opt * dt)
    one_minus_a_full = (1.0 - a_full)
    T_sim = np.empty_like(T)
    T_sim[:] = np.nan
    # Initialize with first observed
    first_idx = idx[0] - 1
    T_sim[first_idx] = T[first_idx]
    # Step through using piecewise-constant Tout and b
    for j, di in enumerate(dt):
        i0 = idx[j] - 1
        i1 = idx[j]
        a_j = a_full[j]
        om_a_j = one_minus_a_full[j]
        base = Tout[i0] + (b_opt / k_opt)
        T_sim[i1] = a_j * T_sim[i0] + om_a_j * base

    # Compute R² over the valid simulated points
    valid = np.isfinite(T_sim)
    obs = T[valid]
    pred = T_sim[valid]
    sse_T = float(np.dot(obs - pred, obs - pred))
    sst_T = float(np.dot(obs - obs.mean(), obs - obs.mean()))
    r2_T = float(1.0 - sse_T / sst_T) if sst_T > 0 else float('nan')

    tau_s = 1.0 / k_opt if k_opt > 0 else float('inf')

    return {
        'k_per_second': k_opt,
        'k_per_hour': k_opt * 3600.0,
        'intercept_degC_per_s': b_opt,
        'tau_seconds': tau_s,
        'tau_hours': tau_s / 3600.0 if np.isfinite(tau_s) else float('inf'),
        'r2_temperature': r2_T,
        'n_samples': int(len(dt)),
        'search_bounds_k_per_s': list(bounds),
        'success': bool(res.success),
        'message': str(res.message),
    }


def main():
    # Sensors
    tout = 'sensor.torild_air_temperature'
    thouse = 'sensor.house_hall_temp'
    pradiator = 'sensor.radiator_power'  # Used only for plotting context

    # Time period (adjust as needed)
    stockholm_tz = ZoneInfo('Europe/Stockholm')

    # Example: cooling-only window
    start_stockholm = datetime(2025, 1, 26, 17, 0, 0).replace(tzinfo=stockholm_tz)
    end_stockholm = datetime(2025, 1, 28, 18, 0, 0).replace(tzinfo=stockholm_tz)

    # Convert to UTC
    start_utc = start_stockholm.astimezone(ZoneInfo('UTC'))
    end_utc = end_stockholm.astimezone(ZoneInfo('UTC'))

    print(f"Loading house data from {start_stockholm.isoformat()} to {end_stockholm.isoformat()} (Stockholm time)")
    print(f"UTC times: {start_utc.isoformat()} to {end_utc.isoformat()}")

    start_ts = start_utc.timestamp()
    end_ts = end_utc.timestamp()

    # Load data
    df = load_data([tout, thouse, pradiator], from_time=start_ts, to_time=end_ts)
    if df.empty:
        raise ValueError("No data found for the specified period")

    print(f"Loaded {len(df)} samples from {df.index[0]} to {df.index[-1]}")

    # Simple smoothed power (for plotting context only)
    radiator_w = df[pradiator].astype(float).fillna(0.0).clip(lower=0.0)
    try:
        df['P_total_smooth'] = radiator_w.rolling('5min', min_periods=1).mean()
    except Exception:
        df['P_total_smooth'] = radiator_w.rolling(5, min_periods=1).mean()

    # Fit tau (time-domain, exact discrete update, closed-form b for each k)
    results = fit_tau_from_cooling(
        df,
        tout_col=tout,
        thouse_col=thouse,
        k_bounds=(1e-7, 1e-3),  # τ in [~2.8 h, ~2780 h]
        max_gap_seconds=None,
    )

    # Simulate with fitted k, b
    idx = df.index
    dt_sec = idx.to_series().diff().dt.total_seconds().fillna(0.0).astype(float).to_numpy()
    tout_vals = df[tout].astype(float).to_numpy()
    t_obs_vals = df[thouse].astype(float).to_numpy()

    t_sim = np.empty_like(t_obs_vals)
    t_sim[0] = t_obs_vals[0]

    k_val = results['k_per_second']
    b_val = results['intercept_degC_per_s']

    for j in range(1, len(t_sim)):
        dTdt = k_val * (tout_vals[j-1] - t_sim[j-1]) + b_val
        t_sim[j] = t_sim[j-1] + dTdt * dt_sec[j]

    df['T_sim'] = t_sim

    # Print summary
    print('\n=== Time constant estimate (cooling-only, time domain) ===')
    print(f"tau: {results['tau_hours']:.2f} h")
    print(f"k: {results['k_per_hour']:.5f} 1/h")
    print(f"intercept b: {results['intercept_degC_per_s']:.3e} °C/s")
    print(f"R² (temperature): {results['r2_temperature']:.4f}")
    print(f"Samples (intervals): {results['n_samples']}")

    # Save JSON
    output = {
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
        'results': results,
    }

    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    period_tag = f"{start_stockholm.date()}_to_{end_stockholm.date()}"
    json_path = output_dir / f"house_tau_{period_tag}.json"
    with json_path.open('w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to {json_path}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    ax.plot(idx, df[tout].astype(float).to_numpy(), label='Outdoor temp (obs)', color='#1f77b4', linewidth=2)
    ax.plot(idx, t_obs_vals, label='House temp (obs)', color='#ff7f0e', linewidth=2)
    ax.plot(idx, t_sim, label='House temp (sim)', color='#2ca02c', linestyle='--', linewidth=2)

    ax2 = ax.twinx()
    ax2.plot(idx, df['P_total_smooth'], label='Radiator power (W, 5-min mean)', color='#6a51a3', linewidth=1.8, alpha=0.7)
    ax2.set_ylabel('Power (W)', fontsize=16)
    ax2.tick_params(axis='y', labelsize=14)

    ax.set_title(
        (
            f"Cooling fit (time domain): τ = {results['tau_hours']:.2f} h, "
            f"k = {results['k_per_hour']:.5f} 1/h, "
            f"R² = {results['r2_temperature']:.4f}, samples = {results['n_samples']}"
        ),
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

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=14)
    plt.tight_layout()

    png_path = output_dir / f"house_tau_{period_tag}.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot to {png_path}")

    try:
        plt.show()
    except Exception:
        plt.close(fig)


if __name__ == '__main__':
    main()
