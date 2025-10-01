import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from load_data import load_data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def fit_k(df, tout_col, tcaravan_col):
    """
    Estimate k in the model dTcaravan/dt = k * (Tout - Tcaravan)

    Contract:
    - Input df indexed by datetime, minutely or similar, with columns tout_col and tcaravan_col (floats)
    - Output dict with k (1/s), k_per_hour (1/h), r2, stderr, n_samples
    """
    df = df.copy()

    # Compute time delta in seconds between samples (Series[float])
    dt_seconds_s = df.index.to_series().diff().dt.total_seconds()

    # Convert to numpy arrays for arithmetic stability
    dt_seconds = np.asarray(dt_seconds_s.astype(float).values, dtype=float)
    dT = np.asarray(df[tcaravan_col].astype(float).diff().values, dtype=float)
    dT_dt = dT / dt_seconds

    # Driving temperature difference as numpy
    x = np.asarray((df[tout_col].astype(float) - df[tcaravan_col].astype(float)).values, dtype=float)

    # Build DataFrame aligned to original index for filtering and drop NaNs from edges
    data = pd.DataFrame({'x': x, 'y': dT_dt, 'dt': dt_seconds}, index=df.index).dropna()

    # Keep only reasonably uniform sampling (30s..120s) to avoid edge artifacts
    data = data[(data['dt'] >= 30) & (data['dt'] <= 120)]

    if len(data) < 10:
        raise ValueError(f"Not enough samples after cleaning to fit k (have {len(data)})")

    x_vals = np.asarray(data['x'].values, dtype=float)
    y_vals = np.asarray(data['y'].values, dtype=float)

    # Fit through origin: minimize ||y - k x||^2 -> k = (x·y)/(x·x)
    denom = np.dot(x_vals, x_vals)
    if denom == 0:
        raise ValueError("Zero variance in (Tout - Tcaravan); cannot estimate k")
    k = float(np.dot(x_vals, y_vals) / denom)

    # Residuals and statistics
    y_hat = k * x_vals
    residuals = y_vals - y_hat
    n = len(y_vals)
    # Degrees of freedom: 1 parameter (slope through origin)
    dof = max(n - 1, 1)
    sse = float(np.dot(residuals, residuals))
    sigma2 = sse / dof
    stderr = float(np.sqrt(sigma2 / denom))
    # R^2 for no-intercept model: 1 - SSE / sum(y^2)
    tss0 = float(np.dot(y_vals, y_vals))
    r2 = float(1.0 - sse / tss0) if tss0 > 0 else float('nan')

    return {
        'k_per_second': k,
        'k_per_hour': k * 3600.0,
        'stderr_per_second': stderr,
        'stderr_per_hour': stderr * 3600.0,
        'r2_no_intercept': r2,
        'n_samples': n,
    }


def main():
    # Sensors
    tout = 'sensor.climate_outdoor_temperature'
    tcaravan = 'sensor.climate_b_kontor_temperature'

    # Time window in UTC as requested
    start_dt = datetime.fromisoformat('2025-09-28T16:00:00+00:00')
    end_dt = datetime.fromisoformat('2025-09-29T04:00:00+00:00')
    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()

    # Load minutely-interpolated data using project helper
    df = load_data([tout, tcaravan], from_time=start_ts, to_time=end_ts)

    # Fit k
    results = fit_k(df, tout_col=tout, tcaravan_col=tcaravan)

    # Print results
    print('=== Caravan heat exchange estimate ===')
    print(f"Time window: {start_dt.isoformat()} to {end_dt.isoformat()} (UTC)")
    print(f"Samples used: {results['n_samples']}")
    print(f"k = {results['k_per_second']:.6e} 1/s  (± {results['stderr_per_second']:.2e})")
    print(f"k = {results['k_per_hour']:.6e} 1/h  (± {results['stderr_per_hour']:.2e})")
    print(f"R^2 (no intercept): {results['r2_no_intercept']:.4f}")

    # Save to output JSON for reproducibility
    out = {
        'start_utc': start_dt.isoformat(),
        'end_utc': end_dt.isoformat(),
        'sensors': {'Tout': tout, 'Tcaravan': tcaravan},
        'results': results,
    }
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / 'caravan_k_2025-09-28T16_to_2025-09-29T04.json'
    with out_path.open('w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved results to {out_path}")

    # Simulate Tcaravan using fitted k and Tout, Euler integration on original grid
    k = results['k_per_second']
    idx = df.index
    dt_sec = idx.to_series().diff().dt.total_seconds().fillna(0.0).astype(float).to_numpy()
    tout_vals = df[tout].astype(float).to_numpy()
    t_obs_vals = df[tcaravan].astype(float).to_numpy()
    t_sim = np.empty_like(t_obs_vals)
    t_sim[0] = t_obs_vals[0]
    for i in range(1, len(t_sim)):
        t_sim[i] = t_sim[i-1] + k * (tout_vals[i-1] - t_sim[i-1]) * dt_sec[i]

    # Plot Tout, Tcaravan (observed), and Tcaravan (simulated)
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    ax.plot(idx, tout_vals, label='Tout (observed)', color='#1f77b4')
    ax.plot(idx, t_obs_vals, label='Tcaravan (observed)', color='#ff7f0e')
    ax.plot(idx, t_sim, label='Tcaravan (simulated)', color='#2ca02c', linestyle='--')
    ax.set_title(f"Caravan model fit: k = {results['k_per_hour']:.3f} 1/h (R²={results['r2_no_intercept']:.3f})")
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    ax.legend(loc='best')
    plt.tight_layout()

    png_path = output_dir / 'caravan_k_timeseries_2025-09-28T16_to_2025-09-29T04.png'
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot to {png_path}")
    try:
        plt.show()
    except Exception:
        # In headless environments, showing may fail; close instead
        plt.close(fig)


if __name__ == '__main__':
    main()
