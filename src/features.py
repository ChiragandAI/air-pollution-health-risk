"""Feature engineering for the air-pollution / health-risk pipeline.

Adds rolling exposure windows, lag features, cyclic day-of-year encoding,
and city one-hot columns. Returns the feature matrix `X`, the classification
target `y_class`, the regression target `y_reg`, and the auxiliary columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


ROLL_WINDOWS = [3, 7]   # short-term and weekly exposure windows
LAG_DAYS = [1, 2]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def add_exposure_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Per-city rolling means and lags of PM2.5 — the 'exposure' features."""
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    g = df.groupby("city", group_keys=False)
    for w in ROLL_WINDOWS:
        df[f"pm25_roll{w}"] = g["pm25"].transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )
    for lag in LAG_DAYS:
        df[f"pm25_lag{lag}"] = g["pm25"].shift(lag).fillna(df["pm25"])
    return df


def build_feature_table(df: pd.DataFrame):
    """Return (X, y_class, y_reg, meta) ready for sklearn."""
    df = add_temporal_features(df)
    df = add_exposure_windows(df)

    city_oh = pd.get_dummies(df["city"], prefix="city", dtype=int)

    feature_cols = [
        # current pollutants
        "pm25", "pm10", "no2", "so2", "co", "o3",
        # weather
        "temp_c", "humidity", "wind_speed", "rainfall_mm",
        # location
        "lat", "lon",
        # temporal cyclic
        "doy_sin", "doy_cos", "is_weekend",
        # exposure windows & lags
        "pm25_roll3", "pm25_roll7", "pm25_lag1", "pm25_lag2",
    ]
    X = pd.concat([df[feature_cols], city_oh], axis=1)
    # Explicit ordered category so .cat.codes are deterministic and match
    # config.RISK_LABELS: Low=0, Moderate=1, High=2. Without this, pandas sorts
    # categories alphabetically (High=0, Low=1, Moderate=2) and the downstream
    # app/report would associate the wrong names with each class.
    y_class = pd.Categorical(
        df["risk_label"], categories=config.RISK_LABELS, ordered=True,
    )
    y_class = pd.Series(y_class, index=df.index, name="risk_label")
    y_reg = df["aqi"]
    meta = df[["date", "city", "lat", "lon"]].copy()
    return X, y_class, y_reg, meta
