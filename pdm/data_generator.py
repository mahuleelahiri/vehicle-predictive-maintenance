"""Synthetic fleet telemetry generator.

Each vehicle produces one aggregated sensor reading per operating hour. A
subset of vehicles develops one of four failure modes; their sensors drift
progressively from a hidden onset point until the failure event at the end of
the run. Healthy vehicles run to the end of the observation window without
failing (right-censored), but still show noise, transient spikes and slow
ageing so the task is not trivially separable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FAILURE_HORIZON, RAW_DATA_PATH, SEED

FAILURE_MODES = ["bearing_wear", "overheating", "oil_system", "electrical"]


def _ar1(n: int, mean: float, phi: float, sigma: float, rng) -> np.ndarray:
    """Autocorrelated driving profile around a mean."""
    x = np.empty(n)
    x[0] = mean
    for t in range(1, n):
        x[t] = mean + phi * (x[t - 1] - mean) + rng.normal(0, sigma)
    return x


def _simulate_vehicle(vid: int, rng) -> pd.DataFrame:
    n = int(rng.integers(300, 800))
    fails = rng.random() < 0.55
    mode = rng.choice(FAILURE_MODES) if fails else "none"

    # Degradation curve d(t) in [0, 1]: zero before onset, convex rise to failure.
    d = np.zeros(n)
    if fails:
        lead = int(rng.integers(70, 220))
        onset = max(n - lead, 1)
        ramp = np.linspace(0, 1, n - onset)
        d[onset:] = ramp ** rng.uniform(1.6, 2.6)

    # Per-vehicle baselines (manufacturing / usage variation).
    route_speed = rng.uniform(35, 85)
    temp_base = rng.normal(88, 2.0)
    oil_base = rng.normal(26, 2.0)
    vib_base = rng.normal(0.9, 0.12)
    batt_base = rng.normal(13.9, 0.12)
    fuel_base = rng.normal(1.8, 0.25)
    start_hours = rng.uniform(200, 9000)
    age = np.linspace(0, 1, n) * rng.uniform(0, 0.15)  # mild ageing

    speed = np.clip(_ar1(n, route_speed, 0.85, 9.0, rng), 0, 140)
    rpm = 750 + speed * rng.uniform(22, 30) + rng.normal(0, 120, n)
    engine_temp = temp_base + 0.04 * speed + rng.normal(0, 1.4, n)
    coolant_temp = engine_temp - rng.uniform(2, 5) + rng.normal(0, 1.0, n)
    oil_pressure = oil_base + 0.006 * rpm + rng.normal(0, 1.8, n)
    vibration = vib_base + 0.00025 * rpm + age + np.abs(rng.normal(0, 0.18, n))
    battery = batt_base + rng.normal(0, 0.12, n) - age
    fuel = fuel_base + 0.075 * speed + rng.normal(0, 0.45, n)

    # Failure-mode specific drift.
    if mode == "bearing_wear":
        vibration += rng.uniform(3.0, 5.0) * d
        engine_temp += 5 * d
        coolant_temp += 3 * d
        fuel += 0.8 * d
    elif mode == "overheating":
        engine_temp += rng.uniform(18, 30) * d
        coolant_temp += rng.uniform(20, 32) * d
        fuel += 1.2 * d
        vibration += 0.4 * d
    elif mode == "oil_system":
        oil_pressure -= rng.uniform(15, 24) * d
        engine_temp += 9 * d
        vibration += 1.2 * d
    elif mode == "electrical":
        battery -= rng.uniform(1.8, 3.0) * d
        rpm += rng.normal(0, 250, n) * d  # erratic idle/ignition
        fuel += 0.6 * d

    # Transient spikes on every vehicle (false-alarm material).
    for arr, scale in ((engine_temp, 10), (vibration, 1.5), (oil_pressure, -8), (battery, -0.8)):
        idx = rng.random(n) < 0.01
        arr[idx] += scale * rng.uniform(0.5, 1.0, idx.sum())

    hours_left = np.arange(n)[::-1].astype(float)
    ttf = np.where(fails, hours_left, np.nan)
    label = (fails & (hours_left <= FAILURE_HORIZON)).astype(int)

    return pd.DataFrame({
        "vehicle_id": f"V{vid:04d}",
        "timestamp": np.arange(n),
        "engine_temp": engine_temp,
        "rpm": np.clip(rpm, 600, None),
        "oil_pressure": np.clip(oil_pressure, 0, None),
        "vibration": np.clip(vibration, 0, None),
        "battery_voltage": battery,
        "coolant_temp": coolant_temp,
        "fuel_consumption": np.clip(fuel, 0.5, None),
        "vehicle_speed": speed,
        "operating_hours": start_hours + np.arange(n),
        "failure_mode": mode,
        "time_to_failure": ttf,
        "failure_within_horizon": label,
    })


def generate_fleet(n_vehicles: int = 300, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.concat([_simulate_vehicle(i, rng) for i in range(n_vehicles)], ignore_index=True)
    num = df.select_dtypes("float").columns.drop("time_to_failure")
    df[num] = df[num].round(3)
    return df


def main(n_vehicles: int = 300) -> pd.DataFrame:
    df = generate_fleet(n_vehicles)
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved {len(df):,} readings from {df.vehicle_id.nunique()} vehicles -> {RAW_DATA_PATH}")
    print(f"Positive rate (fail within {FAILURE_HORIZON}h): {df.failure_within_horizon.mean():.2%}")
    print(df.groupby("vehicle_id").failure_mode.first().value_counts().to_string())
    return df


if __name__ == "__main__":
    main()
