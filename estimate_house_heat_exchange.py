import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from load_data import load_data

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def fit_k_alpha(df, tout_col: str, thouse_col: str, ptotal_col: str):
    """
    Estimate parameters in the model:
        dT/dt = k * (Tout - Thouse) + alpha * P_total

    Physical interpretation (first-order RC model):
        C * dT/dt = (Tout - Thouse)/R + P_total
      => dT/dt = (Tout - Thouse)/(R*C) + P_total/C
      with k = 1/(R*C) [1/s] and alpha = 1/C [°C/(W*s)]

    Where P_total = radiator_power + constant_appliances_power.

    Returns dict with primary physical parameters R [°C/W] and C [J/°C],
    along with k, alpha, statistics, and samples used.
    """
    df = df.copy()

    # Compute time delta in seconds between samples (Series[float])
    dt_seconds_s = df.index.to_series().diff().dt.total_seconds()

    # Convert to numpy arrays for arithmetic stability
    dt_seconds = np.asarray(dt_seconds_s.astype(float).values, dtype=float)
    dT = np.asarray(df[thouse_col].astype(float).diff().values, dtype=float)
    dT_dt = dT / dt_seconds

    # Predictors
    x1 = np.asarray((df[tout_col].astype(float) - df[thouse_col].astype(float)).values, dtype=float)
    x2 = np.asarray(df[ptotal_col].astype(float).values, dtype=float)

    # Build DataFrame aligned to original index for filtering and drop NaNs from edges
    data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': dT_dt, 'dt': dt_seconds}, index=df.index).dropna()

    # Keep only reasonably uniform sampling (30s..120s) to avoid edge artifacts
    # This is critical when combining multiple time periods with gaps
    data_before_filter = len(data)
    data = data[(data['dt'] >= 30) & (data['dt'] <= 120)]
    data_after_filter = len(data)
    
    print(f"Filtering: kept {data_after_filter}/{data_before_filter} samples ({data_before_filter - data_after_filter} boundary samples excluded)")

    if len(data) < 10:
        raise ValueError(f"Not enough samples after cleaning to fit k (have {len(data)})")

    X = np.column_stack([
        np.asarray(data['x1'].values, dtype=float),
        np.asarray(data['x2'].values, dtype=float)
    ])
    y_vals = np.asarray(data['y'].values, dtype=float)

    # Solve least squares without intercept
    coeffs, residuals_arr, rank, s = np.linalg.lstsq(X, y_vals, rcond=None)
    k = float(coeffs[0])              # 1/(R*C)  [1/s]
    alpha = float(coeffs[1])          # 1/C      [°C/(W*s)]

    # Residuals and statistics
    y_hat = X @ coeffs
    residuals = y_vals - y_hat
    n = len(y_vals)
    p = X.shape[1]
    # Degrees of freedom: p parameters (k, alpha), no intercept
    dof = max(n - p, 1)
    sse = float(np.dot(residuals, residuals))
    sigma2 = sse / dof
    # Standard errors for coefficients
    XtX_inv = np.linalg.pinv(X.T @ X)
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    stderr_k = float(se[0])
    stderr_alpha = float(se[1])

    # Derive physical parameters R [°C/W], C [J/°C], and time constant tau [s]
    # Relations: C = 1/alpha, R = alpha / k, tau = 1/k
    C = (1.0 / alpha) if alpha != 0 else float('inf')
    R = (alpha / k) if (k != 0) else float('inf')
    tau_s = (1.0 / k) if k != 0 else float('inf')

    # Uncertainty propagation (first-order/linearized)
    # C = 1/alpha  => dC/dalpha = -1/alpha^2
    var_alpha = float(cov[1, 1])
    var_k = float(cov[0, 0])
    cov_k_alpha = float(cov[0, 1])
    var_C = (var_alpha / (alpha**4)) if np.isfinite(C) else float('inf')
    # R = alpha/k => dR/dalpha = 1/k, dR/dk = -alpha/k^2
    if np.isfinite(R):
        dR_dalpha = 1.0 / k
        dR_dk = -alpha / (k**2)
        var_R = (dR_dalpha**2) * var_alpha + (dR_dk**2) * var_k + 2.0 * dR_dalpha * dR_dk * cov_k_alpha
    else:
        var_R = float('inf')
    # tau = 1/k => dtau/dk = -1/k^2
    if np.isfinite(tau_s):
        dtau_dk = -1.0 / (k**2)
        var_tau = (dtau_dk**2) * var_k
    else:
        var_tau = float('inf')
    stderr_C = float(np.sqrt(var_C)) if np.isfinite(var_C) else float('inf')
    stderr_R = float(np.sqrt(var_R)) if np.isfinite(var_R) else float('inf')
    stderr_tau_s = float(np.sqrt(var_tau)) if np.isfinite(var_tau) else float('inf')

    # R^2 for no-intercept model: 1 - SSE / sum(y^2) (no intercept definition)
    tss0 = float(np.dot(y_vals, y_vals))
    r2 = float(1.0 - sse / tss0) if tss0 > 0 else float('nan')

    return {
        # Primary physical parameters
        'R_degC_per_W': R,
        'C_J_per_degC': C,
        'tau_seconds': tau_s,
        'tau_hours': tau_s / 3600.0 if np.isfinite(tau_s) else float('inf'),
        'stderr_R_degC_per_W': stderr_R,
        'stderr_C_J_per_degC': stderr_C,
        'stderr_tau_seconds': stderr_tau_s,
        'stderr_tau_hours': (stderr_tau_s / 3600.0) if np.isfinite(stderr_tau_s) else float('inf'),

        # Fitted linear coefficients (for reference/back-compat)
        'k_per_second': k,
        'k_per_hour': k * 3600.0,
        'alpha_per_watt': alpha,  # (deg C / (W*s)) == 1/C
        'capacity_J_per_degC': (1.0 / alpha) if alpha != 0 else float('inf'),
        'stderr_k_per_second': stderr_k,
        'stderr_k_per_hour': stderr_k * 3600.0,
        'stderr_alpha_per_watt': stderr_alpha,

        # Fit quality
        'r2_no_intercept': r2,
        'n_samples': n,
    }


def main():
    # Sensors
    tout = 'sensor.torild_air_temperature'
    thouse = 'sensor.house_hall_temp'
    pradiator = 'sensor.radiator_power'
    
    # Constant appliances power (W)
    constant_appliances_power = 300.0

    # Time period: January 25, 2025 at 00:00 to February 3, 2025 at 22:00, Stockholm timezone
    stockholm_tz = ZoneInfo('Europe/Stockholm')

    # kläppen
    # start_stockholm = datetime(2025, 1, 26, 9, 0, 0).replace(tzinfo=stockholm_tz)
    # end_stockholm = datetime(2025, 2, 2, 17, 0, 0).replace(tzinfo=stockholm_tz)
    
    
    start_stockholm = datetime(2025, 3, 10, 0, 0, 0).replace(tzinfo=stockholm_tz)
    end_stockholm = datetime(2025, 3, 17, 0, 0, 0).replace(tzinfo=stockholm_tz)
    
   
    # Convert to UTC for data loading
    start_utc = start_stockholm.astimezone(ZoneInfo('UTC'))
    end_utc = end_stockholm.astimezone(ZoneInfo('UTC'))
    
    print(f"Loading house data from {start_stockholm.isoformat()} to {end_stockholm.isoformat()} (Stockholm time)")
    print(f"UTC times: {start_utc.isoformat()} to {end_utc.isoformat()}")

    start_ts = start_utc.timestamp()
    end_ts = end_utc.timestamp()

    # Load data for the entire period
    df = load_data([tout, thouse, pradiator], from_time=start_ts, to_time=end_ts)
    if df.empty:
        raise ValueError("No data found for the specified period")

    print(f"Loaded {len(df)} samples from {df.index[0]} to {df.index[-1]}")

    # Compute total power (radiator + constant appliances)
    radiator_w = df[pradiator].astype(float).fillna(0.0).clip(lower=0.0)
    df['P_total'] = radiator_w + constant_appliances_power
    
    # Create smoothed power for plotting
    try:
        df['P_total_smooth'] = df['P_total'].rolling('5min', min_periods=1).mean()
    except Exception:
        df['P_total_smooth'] = df['P_total'].rolling(5, min_periods=1).mean()

    # Fit the model
    results = fit_k_alpha(df, tout_col=tout, thouse_col=thouse, ptotal_col='P_total')

    # Simulate the model
    idx = df.index
    dt_sec = idx.to_series().diff().dt.total_seconds().fillna(0.0).astype(float).to_numpy()
    tout_vals = df[tout].astype(float).to_numpy()
    t_obs_vals = df[thouse].astype(float).to_numpy()
    p_total_vals = df['P_total'].astype(float).to_numpy()
    
    t_sim = np.empty_like(t_obs_vals)
    t_sim[0] = t_obs_vals[0]  # Initialize with first observed temperature
    
    # Use physical parameters R and C for simulation
    R_val = results['R_degC_per_W']
    C_val = results['C_J_per_degC']

    print(f"R (°C/W): {R_val}, C (J/°C): {C_val}, tau (h): {results['tau_hours']}")

    for j in range(1, len(t_sim)):
        dTdt = ( (tout_vals[j-1] - t_sim[j-1]) / (R_val * C_val) ) + (p_total_vals[j-1] / C_val)
        t_sim[j] = t_sim[j-1] + dTdt * dt_sec[j]
    
    df['T_sim'] = t_sim

    # Print results
    print('\n=== House heat exchange estimate (RC model) ===')
    print(f"R: {results['R_degC_per_W']:.3f} ± {results['stderr_R_degC_per_W']:.3f} °C/W")
    print(f"C: {results['C_J_per_degC']:.0f} ± {results['stderr_C_J_per_degC']:.0f} J/°C")
    print(f"tau: {results['tau_hours']:.2f} ± {results['stderr_tau_hours']:.2f} h")
    print(f"R²: {results['r2_no_intercept']:.4f}")
    print(f"Number of samples: {results['n_samples']}")
    print(f"Constant appliances power: {constant_appliances_power} W")

    # Save results to JSON
    output_data = {
        'period': {
            'start_stockholm': start_stockholm.isoformat(),
            'end_stockholm': end_stockholm.isoformat(),
            'start_utc': start_utc.isoformat(),
            'end_utc': end_utc.isoformat()
        },
        'sensors': {
            'Tout': tout,
            'Thouse': thouse, 
            'Pradiator': pradiator
        },
        'constant_appliances_power_W': constant_appliances_power,
        'results': results
    }
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    # Build descriptive filenames using actual period and RC notation
    period_tag = f"{start_stockholm.date()}_to_{end_stockholm.date()}"
    out_path = output_dir / f"house_RC_{period_tag}.json"
    
    with out_path.open('w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    
    # Plot temperatures
    ax.plot(idx, tout_vals, label='Outdoor temp (observed)', color='#1f77b4', linewidth=2)
    ax.plot(idx, t_obs_vals, label='House temp (observed)', color='#ff7f0e', linewidth=2)
    ax.plot(idx, t_sim, label='House temp (simulated)', color='#2ca02c', linestyle='--', linewidth=2)

    # Secondary axis for power
    ax2 = ax.twinx()
    ax2.plot(idx, df['P_total_smooth'], label='Total power (W, 5-min mean)', color='#6a51a3', linewidth=1.8, alpha=0.7)
    ax2.set_ylabel('Power (W)', fontsize=16)
    ax2.tick_params(axis='y', labelsize=14)
    
    # Set title
    ax.set_title(
        f"House thermal model fit: R = {results['R_degC_per_W']:.3f} °C/W, C = {results['C_J_per_degC']:.0f} J/°C, "
        f"τ = {results['tau_hours']:.2f} h, R² = {results['r2_no_intercept']:.4f}, samples = {results['n_samples']}",
        fontsize=18,
    )
    ax.set_xlabel('Time (Stockholm timezone)', fontsize=16)
    ax.set_ylabel('Temperature (°C)', fontsize=16)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    
    ax.tick_params(axis='x', rotation=45, labelsize=12, which='major')
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(True, alpha=0.3)
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=14)
    plt.tight_layout()

    # Save plot
    png_path = output_dir / f"house_RC_{period_tag}.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot to {png_path}")
    
    try:
        plt.show()
    except Exception:
        # In headless environments, showing may fail; close instead
        plt.close(fig)


if __name__ == '__main__':
    main()