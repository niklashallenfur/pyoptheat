from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo

from load_data import load_data


@dataclass
class TimeRange:
    start_local: datetime
    end_local: datetime
    tz: ZoneInfo

    @property
    def start_utc_ts(self) -> float:
        return self.start_local.astimezone(ZoneInfo("UTC")).timestamp()

    @property
    def end_utc_ts(self) -> float:
        return self.end_local.astimezone(ZoneInfo("UTC")).timestamp()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute average hourly electricity consumption profile by clock hour "
            "from a cumulative kWh Home Assistant meter."
        )
    )
    p.add_argument(
        "--entity",
        default="sensor.electricity_meter_energy_consumption_total",
        help="Home Assistant sensor entity_id for cumulative energy (kWh).",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--days",
        type=int,
        default=60,
        help="Number of days back from now to include (default: 60).",
    )
    group.add_argument(
        "--start",
        type=str,
        help="Start time in ISO 8601 (e.g., 2025-08-21T00:00:00 or 2025-08-21T00:00:00+02:00).",
    )
    p.add_argument(
        "--end",
        type=str,
        help="End time in ISO 8601 (default: now).",
    )
    p.add_argument(
        "--tz",
        default="Europe/Stockholm",
        help="Time zone used for clock hours (default: Europe/Stockholm).",
    )
    p.add_argument(
        "--save-png",
        type=str,
        help="Optional path to save the plot as PNG.",
    )
    p.add_argument(
        "--save-csv",
        type=str,
        help="Optional path to save the averaged 24-hour profile as CSV.",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title override.",
    )
    return p.parse_args()


def parse_time_range(args: argparse.Namespace) -> TimeRange:
    tz = ZoneInfo(args.tz)

    def parse_iso_local(s: str) -> datetime:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        return dt.astimezone(tz)

    if args.start:
        start_local = parse_iso_local(args.start)
        end_local = parse_iso_local(args.end) if args.end else datetime.now(tz)
    else:
        # last N days window
        end_local = datetime.now(tz)
        start_local = end_local - timedelta(days=args.days)

    # Normalize to exact hour boundaries for clean hour-of-day grouping
    start_local = start_local.replace(minute=0, second=0, microsecond=0)
    # End is exclusive; include the last hour boundary by snapping up to the next hour
    if end_local.minute or end_local.second or end_local.microsecond:
        end_local = (end_local.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))

    return TimeRange(start_local=start_local, end_local=end_local, tz=tz)


def compute_hourly_profile(
    df: pd.DataFrame, entity: str, tz: ZoneInfo
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Convert cumulative kWh readings into hourly kWh consumption in local time, then
    compute the mean/std profile per clock hour (0..23).

    Returns:
      - avg_by_hour: Series indexed by 0..23 with mean kWh per hour
      - std_by_hour: Series indexed by 0..23 with std kWh per hour
      - hourly_kwh: hourly series (localized) for reference and debugging
    """
    if df.empty:
        raise ValueError("No data loaded in the selected time range.")

    # load_data returns a naive datetime index assumed to be in local system time.
    # Localize to provided tz (do not convert), so hour-of-day grouping is correct.
    s = df[entity].astype(float).sort_index()
    if s.index.tz is None:
        s.index = s.index.tz_localize(tz)
    else:
        s = s.tz_convert(tz)

    # Build hourly cumulative series by forward-filling the last known cumulative kWh
    hourly_cum = s.resample("H").ffill()

    # Compute hourly kWh by differencing cumulative readings at hour boundaries
    hourly_kwh = hourly_cum.diff()

    # Handle resets or negative diffs by clipping to zero
    hourly_kwh = hourly_kwh.clip(lower=0)

    # Drop first NaN and any remaining NaNs
    hourly_kwh = hourly_kwh.dropna()

    # Compute average and std dev by hour-of-day
    hours = hourly_kwh.index.hour
    avg_by_hour = hourly_kwh.groupby(hours).mean()
    std_by_hour = hourly_kwh.groupby(hours).std(ddof=1)

    # Ensure full 0..23 index exists even if some hours are missing
    full_hours = pd.RangeIndex(0, 24)
    avg_by_hour = avg_by_hour.reindex(full_hours)
    std_by_hour = std_by_hour.reindex(full_hours)

    return avg_by_hour, std_by_hour, hourly_kwh


def plot_profile(
    avg_by_hour: pd.Series, std_by_hour: pd.Series, tr: TimeRange, title: Optional[str] = None
):
    x = np.arange(24)
    y = avg_by_hour.values
    yerr = std_by_hour.values

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, marker="o", linewidth=2, color="#1f77b4", label="Average kWh per hour")
    if not np.all(np.isnan(yerr)):
        ax.fill_between(x, y - yerr, y + yerr, color="#1f77b4", alpha=0.15, label="±1σ")

    ax.set_xticks(x)
    ax.set_xlabel("Clock hour (local)")
    ax.set_ylabel("Energy consumption (kWh)")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"Average hourly consumption profile by clock hour\n"
            f"{tr.start_local.strftime('%Y-%m-%d')} to {tr.end_local.strftime('%Y-%m-%d')} ({tr.tz.key})"
        )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def main():
    args = parse_args()
    tr = parse_time_range(args)

    print(
        f"Loading '{args.entity}' from {tr.start_local.isoformat()} to {tr.end_local.isoformat()} ({tr.tz.key})"
    )
    print(
        f"UTC range: {datetime.fromtimestamp(tr.start_utc_ts, ZoneInfo('UTC'))} to "
        f"{datetime.fromtimestamp(tr.end_utc_ts, ZoneInfo('UTC'))}"
    )

    # Load via project helper
    df = load_data([args.entity], from_time=tr.start_utc_ts, to_time=tr.end_utc_ts)
    if df.empty:
        print("No data returned from load_data() for the selected range.")
        return

    print(f"Loaded {len(df)} samples from {df.index.min()} to {df.index.max()}")

    # Compute profile
    avg_by_hour, std_by_hour, hourly_kwh = compute_hourly_profile(df, args.entity, tr.tz)

    # Quick stats
    daily_avg_kwh = float(avg_by_hour.sum(skipna=True))
    print(f"Average daily consumption (sum of hourly means): {daily_avg_kwh:.2f} kWh/day")

    # Print 24-value array (average kWh by clock hour 0..23)
    avg_array = [round(float(x), 3) if pd.notna(x) else None for x in avg_by_hour.values]
    print(avg_array)

    # Plot
    fig, ax = plot_profile(avg_by_hour, std_by_hour, tr, title=args.title)

    # Optional save artifacts
    if args.save_csv:
        out_df = pd.DataFrame(
            {
                "hour": np.arange(24),
                "avg_kwh": avg_by_hour.values,
                "std_kwh": std_by_hour.values,
            }
        )
        out_df.to_csv(args.save_csv, index=False)
        print(f"Saved hourly profile CSV to {args.save_csv}")

    if args.save_png:
        fig.savefig(args.save_png, dpi=150)
        print(f"Saved plot PNG to {args.save_png}")

    try:
        plt.show()
    except Exception:
        plt.close(fig)


if __name__ == "__main__":
    main()
