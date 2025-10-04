import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
import sys
from pathlib import Path

# Add parent directory to path to import load_data
sys.path.append(str(Path(__file__).parent.parent))
from load_data import load_data

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def fit_k_alpha(df, tout_col: str, tcaravan_col: str, ptotal_col: str):
    """
    Estimate parameters in the model:
        dT/dt = k * (Tout - Tcaravan) + alpha * P_total

    Where P_total = heater_power + human_heat (assumed constant 100W already added).

    Returns dict with k, alpha, capacity (=1/alpha), statistics, and samples used.
    """
    df = df.copy()

    # Compute time delta in seconds between samples (Series[float])
    dt_seconds_s = df.index.to_series().diff().dt.total_seconds()

    # Convert to numpy arrays for arithmetic stability
    dt_seconds = np.asarray(dt_seconds_s.astype(float).values, dtype=float)
    dT = np.asarray(df[tcaravan_col].astype(float).diff().values, dtype=float)
    dT_dt = dT / dt_seconds

    # Predictors
    x1 = np.asarray((df[tout_col].astype(float) - df[tcaravan_col].astype(float)).values, dtype=float)
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
    k = float(coeffs[0])
    alpha = float(coeffs[1])

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
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    stderr_k = float(se[0])
    stderr_alpha = float(se[1])

    # R^2 for no-intercept model: 1 - SSE / sum(y^2) (no intercept definition)
    tss0 = float(np.dot(y_vals, y_vals))
    r2 = float(1.0 - sse / tss0) if tss0 > 0 else float('nan')

    return {
        'k_per_second': k,
        'k_per_hour': k * 3600.0,
        'alpha_per_watt': alpha,  # (deg C / (W*s))
        'capacity_J_per_degC': (1.0 / alpha) if alpha != 0 else float('inf'),
        'stderr_k_per_second': stderr_k,
        'stderr_k_per_hour': stderr_k * 3600.0,
        'stderr_alpha_per_watt': stderr_alpha,
        'r2_no_intercept': r2,
        'n_samples': n,
    }


def generate_time_periods(start_date: str, end_date: str) -> List[Tuple[datetime, datetime]]:
    """
    Generate time periods from start_date to end_date.
    Each period is from 19:00 Swedish time to 06:00 the next day in Swedish time.
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD' 
    
    Returns:
        List of (start_datetime_utc, end_datetime_utc) tuples
    """
    swedish_tz = ZoneInfo('Europe/Stockholm')
    periods = []
    
    current_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    while current_date <= end_date_obj:
        # Create start time: 19:00 Swedish time on current_date
        start_swedish = datetime.combine(current_date, datetime.min.time().replace(hour=19)).replace(tzinfo=swedish_tz)
        
        # Create end time: 06:00 Swedish time on the next day  
        end_swedish = datetime.combine(current_date + timedelta(days=1), datetime.min.time().replace(hour=6)).replace(tzinfo=swedish_tz)
        
        # Convert to UTC for data loading
        start_utc = start_swedish.astimezone(ZoneInfo('UTC'))
        end_utc = end_swedish.astimezone(ZoneInfo('UTC'))
        
        periods.append((start_utc, end_utc))
        current_date += timedelta(days=1)
    
    return periods


def main():
    # Sensors
    tout = 'sensor.climate_outdoor_temperature'
    tcaravan = 'sensor.climate_b_kontor_temperature'
    pheater = 'sensor.plug_office_power'

    # Generate multiple time periods from 2025-09-25 to 2025-10-03
    # Each period: 19:00 Swedish time to 06:00 next day Swedish time
    periods = generate_time_periods('2025-09-25', '2025-10-03')
    
    print(f"Loading data for {len(periods)} periods...")

    # Fit and simulate per-night; also build combined df for plotting
    night_results: List[Dict[str, Any]] = []
    per_night_frames: List[pd.DataFrame] = []

    for i, (start_utc, end_utc) in enumerate(periods):
        print(f"  Period {i+1}: {start_utc.isoformat()} to {end_utc.isoformat()} (UTC)")
        start_ts = start_utc.timestamp()
        end_ts = end_utc.timestamp()

        # Load data for this night
        df_period = load_data([tout, tcaravan, pheater], from_time=start_ts, to_time=end_ts)
        if df_period.empty:
            print(f"    Warning: No data found for period {i+1}")
            continue

        # Compute total power (cap heater) + 100W humans
        heater_w = df_period[pheater].astype(float).fillna(0.0).clip(lower=0.0, upper=2000.0)
        df_period['P_total'] = heater_w + 100.0
        try:
            df_period['P_total_smooth'] = df_period['P_total'].rolling('5min', min_periods=1).mean()
        except Exception:
            df_period['P_total_smooth'] = df_period['P_total'].rolling(5, min_periods=1).mean()

        # Fit per-night
        res = fit_k_alpha(df_period, tout_col=tout, tcaravan_col=tcaravan, ptotal_col='P_total')

        # Simulate per-night with reset initial condition
        idx = df_period.index
        dt_sec = idx.to_series().diff().dt.total_seconds().fillna(0.0).astype(float).to_numpy()
        tout_vals = df_period[tout].astype(float).to_numpy()
        t_obs_vals = df_period[tcaravan].astype(float).to_numpy()
        p_total_vals = df_period['P_total'].astype(float).to_numpy()
        t_sim = np.empty_like(t_obs_vals)
        t_sim[0] = t_obs_vals[0]
        k_val = res['k_per_second']
        alpha_val = res['alpha_per_watt']
        for j in range(1, len(t_sim)):
            t_sim[j] = (
                t_sim[j-1]
                + (k_val * (tout_vals[j-1] - t_sim[j-1]) + alpha_val * p_total_vals[j-1]) * dt_sec[j]
            )
        df_period['T_sim'] = t_sim

        # Record results
        night_results.append({
            'night_index': i + 1,
            'start_utc': start_utc.isoformat(),
            'end_utc': end_utc.isoformat(),
            'start_swedish': start_utc.astimezone(ZoneInfo('Europe/Stockholm')).isoformat(),
            'end_swedish': end_utc.astimezone(ZoneInfo('Europe/Stockholm')).isoformat(),
            **res,
        })

        per_night_frames.append(df_period)

    if not per_night_frames:
        raise ValueError("No data found for any of the specified periods")

    # Combine all nights for plotting/inspection
    df = pd.concat(per_night_frames, axis=0).sort_index()
    print(f"Combined data: {len(df)} samples from {df.index[0]} to {df.index[-1]}")

    # Verify slicing by checking hours
    hours = [t.hour for t in df.index]
    print(f"Hours present in data: {sorted(set(hours))}")
    print(f"Expected hours: 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6")

    # Print per-night summary
    print('=== Caravan heat exchange estimate (per night) ===')
    for nr in night_results:
        print(
            f"Night {nr['night_index']:2d} | k: {nr['k_per_hour']:.3f} 1/h (± {nr['stderr_k_per_hour']:.3f})  "
            f"alpha: {nr['alpha_per_watt']:.3e} °C/(W*s)  R²: {nr['r2_no_intercept']:.3f}  samples: {nr['n_samples']}"
        )
    # Simple averages (optional)
    k_hours = [nr['k_per_hour'] for nr in night_results]
    alphas = [nr['alpha_per_watt'] for nr in night_results]
    print(f"Averages -> k: {np.mean(k_hours):.3f} 1/h, alpha: {np.mean(alphas):.3e} °C/(W*s)")

    # Save to output JSON for reproducibility
    period_info = [
        {
            'start_utc': start_utc.isoformat(), 
            'end_utc': end_utc.isoformat(),
            'start_swedish': start_utc.astimezone(ZoneInfo('Europe/Stockholm')).isoformat(),
            'end_swedish': end_utc.astimezone(ZoneInfo('Europe/Stockholm')).isoformat()
        } 
        for start_utc, end_utc in periods
    ]
    
    out = {
        'date_range': '2025-09-25 to 2025-10-03',
        'period_description': '19:00 to 06:00 next day (Swedish time) for each date',
        'periods': period_info,
        'sensors': {'Tout': tout, 'Tcaravan': tcaravan, 'Pheater': pheater},
        'night_results': night_results,
        'summary': {
            'avg_k_per_hour': float(np.mean([nr['k_per_hour'] for nr in night_results])),
            'avg_alpha_per_watt': float(np.mean([nr['alpha_per_watt'] for nr in night_results])),
            'n_nights': len(night_results),
            'total_samples': int(np.sum([nr['n_samples'] for nr in night_results]))
        }
    }
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / 'caravan_k_multi_period_2025-09-25_to_2025-10-03.json'
    with out_path.open('w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved results to {out_path}")

    # For plotting we already have per-night simulation in df['T_sim']
    idx = df.index
    tout_vals = df[tout].astype(float).to_numpy()
    t_obs_vals = df[tcaravan].astype(float).to_numpy()
    p_total_vals_smooth = df['P_total_smooth'].astype(float).to_numpy()
    t_sim = df['T_sim'].astype(float).to_numpy()

    # Plot Tout, Tcaravan (observed), and Tcaravan (simulated)
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    ax.plot(idx, tout_vals, label='Tout (observed)', color='#1f77b4', linewidth=2)
    ax.plot(idx, t_obs_vals, label='Tcaravan (observed)', color='#ff7f0e', linewidth=2)
    ax.plot(idx, t_sim, label='Tcaravan (simulated)', color='#2ca02c', linestyle='--', linewidth=2)

    # Secondary axis for power
    ax2 = ax.twinx()
    ax2.plot(idx, p_total_vals_smooth, label='P_total (W, 5-min mean)', color='#6a51a3', linewidth=1.8, alpha=0.7)
    ax2.set_ylabel('Power (W)', fontsize=16)
    ax2.tick_params(axis='y', labelsize=14)
    
    # Add vertical lines to show period boundaries (19:00 and 06:00)
    for start_utc, end_utc in periods:
        # Convert to the same timezone as the data for plotting
        start_local = start_utc.astimezone(ZoneInfo('Europe/Stockholm'))
        end_local = end_utc.astimezone(ZoneInfo('Europe/Stockholm'))
        # Shade night period
        if getattr(idx, 'tz', None) is None:
            s_span = start_local.replace(tzinfo=None)
            e_span = end_local.replace(tzinfo=None)
        else:
            s_span = start_local
            e_span = end_local
        ax.axvspan(s_span, e_span, color='gray', alpha=0.08, zorder=0)
        # Use s_span/e_span for lines as well to ensure matching tz-awareness with the x data
        ax.axvline(s_span, color='red', linestyle=':', alpha=0.7, linewidth=1)
        ax.axvline(e_span, color='red', linestyle=':', alpha=0.7, linewidth=1)

    # Title with aggregated metrics (avoid undefined 'results')
    avg_k_h = float(np.mean([nr['k_per_hour'] for nr in night_results]))
    avg_r2 = float(np.mean([nr['r2_no_intercept'] for nr in night_results]))
    ax.set_title(
        f"Caravan model fit (19:00–06:00, per-night simulation): avg k = {avg_k_h:.3f} 1/h (avg R²={avg_r2:.3f}), nights = {len(night_results)}",
        fontsize=20,
    )
    ax.set_xlabel('Time (Swedish timezone)', fontsize=16)
    ax.set_ylabel('Temperature (°C)', fontsize=16)
    
    # Set x-axis to show hourly ticks
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    
    ax.tick_params(axis='x', rotation=45, labelsize=12, which='major')
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(True, alpha=0.3)
    # Merge legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=14)
    plt.tight_layout()

    png_path = output_dir / 'caravan_k_multi_period_2025-09-25_to_2025-10-03.png'
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot to {png_path}")
    try:
        plt.show()
    except Exception:
        # In headless environments, showing may fail; close instead
        plt.close(fig)


if __name__ == '__main__':
    main()
