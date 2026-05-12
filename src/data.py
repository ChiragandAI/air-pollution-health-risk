"""Synthetic multi-city air-quality + weather dataset generator.

The shape of the data mirrors a typical environmental-health modelling table:
one row per (city, date) with pollutant concentrations, weather, location, and
a derived health-risk label. The generator is deterministic given the seed in
config.py so the rest of the pipeline is fully reproducible.

Replace `generate()` with a real loader (CPCB / OpenAQ / NOAA / etc.) to plug
the same downstream pipeline into real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _seasonal(day_of_year: np.ndarray, amp: float, phase: float = 0.0) -> np.ndarray:
    """Sinusoidal seasonal component peaking around `phase` (day-of-year)."""
    return amp * np.cos(2 * np.pi * (day_of_year - phase) / 365.0)


def generate(n_days: int = 365, start: str = "2024-01-01") -> pd.DataFrame:
    """Generate a synthetic AIRCARE-style dataset.

    Each city gets `n_days` daily observations starting at `start`. PM2.5 is
    driven by a city baseline + winter-peak seasonality (worst in Dec–Jan) +
    a wind/rain suppression effect + weekday/weekend variation + noise. Other
    pollutants and weather variables are correlated with PM2.5 to mimic real
    environmental coupling.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()

    rows: list[pd.DataFrame] = []
    for city in config.CITIES:
        # PM2.5: baseline + winter peak (phase=15) + weekend dip + weather + noise
        winter_peak = _seasonal(doy, amp=city["pm25_base"] * 0.45, phase=15.0)
        wind_speed = np.clip(rng.normal(2.5, 1.2, n_days), 0.2, 8.0)
        rainfall = np.clip(rng.exponential(1.5, n_days) * (1 + _seasonal(doy, 0.6, 200)), 0, 60)
        humidity = np.clip(rng.normal(60, 15, n_days)
                           + _seasonal(doy, 12.0, 200), 15, 100)
        temp = (22.0
                + _seasonal(doy, amp=8.0, phase=200)        # summer peak
                + rng.normal(0, 2.0, n_days)
                - 0.05 * (humidity - 60))
        weekend_dip = np.where(dow >= 5, -city["pm25_base"] * 0.08, 0.0)
        weather_effect = -6.0 * wind_speed - 0.8 * rainfall
        pm25 = (city["pm25_base"]
                + winter_peak
                + weekend_dip
                + weather_effect
                + rng.normal(0, city["pm25_base"] * 0.15, n_days))
        pm25 = np.clip(pm25, 5.0, 600.0)

        # Other pollutants — loose linear+noise relationships to PM2.5
        pm10 = np.clip(pm25 * rng.uniform(1.4, 1.9, n_days) + rng.normal(0, 8, n_days), 8, 800)
        no2  = np.clip(0.35 * pm25 + rng.normal(15, 5, n_days), 2, 200)
        so2  = np.clip(0.10 * pm25 + rng.normal(8, 3, n_days), 1, 100)
        co   = np.clip(0.020 * pm25 + rng.normal(0.8, 0.3, n_days), 0.1, 8)
        o3   = np.clip(40 + 0.3 * temp - 0.10 * pm25 + rng.normal(0, 8, n_days), 5, 180)

        # AQI: simple linear surrogate of PM2.5 (real AQI is piecewise; this is a stand-in)
        aqi = np.clip(1.6 * pm25 + 0.2 * pm10 + rng.normal(0, 10, n_days), 10, 800)

        df = pd.DataFrame({
            "date": dates,
            "city": city["city"],
            "lat": city["lat"],
            "lon": city["lon"],
            "pm25": pm25.round(2),
            "pm10": pm10.round(2),
            "no2": no2.round(2),
            "so2": so2.round(2),
            "co": co.round(3),
            "o3": o3.round(2),
            "temp_c": temp.round(2),
            "humidity": humidity.round(1),
            "wind_speed": wind_speed.round(2),
            "rainfall_mm": rainfall.round(2),
            "aqi": aqi.round(1),
        })
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)

    # Composite health-risk score: PM2.5 dominates, but PM10, NO2, humidity, and a
    # noise term also contribute. Binning this composite produces labels that are
    # *correlated with* PM2.5 (as in reality) without being a pure function of it,
    # so the classifier has something non-trivial to learn.
    score_rng = np.random.default_rng(config.RANDOM_SEED + 1)
    composite = (
        out["pm25"]
        + 0.35 * out["pm10"]
        + 0.45 * out["no2"]
        + 0.15 * (out["humidity"] - 60.0)
        - 1.2 * out["wind_speed"]
        + score_rng.normal(0.0, 18.0, size=len(out))   # irreducible noise
    )
    # Threshold the composite at empirical 60th / 88th percentiles so the class
    # balance is broadly Low > Moderate > High (similar to real exposure data).
    q1, q2 = np.quantile(composite, [0.60, 0.88])
    out["risk_label"] = pd.cut(
        composite, bins=[-np.inf, q1, q2, np.inf], labels=config.RISK_LABELS,
    ).astype(str)
    return out


def load_or_generate() -> pd.DataFrame:
    """Load the cached synthetic dataset, generating it once if it doesn't exist."""
    if config.DATA_CSV.exists():
        df = pd.read_csv(config.DATA_CSV, parse_dates=["date"])
        return df
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(config.DATA_CSV, index=False)
    return df


if __name__ == "__main__":
    df = load_or_generate()
    print(df.head())
    print(f"\nshape={df.shape}  cities={df['city'].nunique()}  "
          f"date range={df['date'].min().date()} → {df['date'].max().date()}")
    print("\nRisk-label distribution:")
    print(df["risk_label"].value_counts())
