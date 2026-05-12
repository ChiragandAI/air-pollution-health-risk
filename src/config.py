"""Paths, constants, and risk thresholds for the air-pollution pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "synthetic"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"

DATA_CSV = DATA_DIR / "aircare_synthetic.csv"

# Indian cities with rough lat/lon and a baseline PM2.5 level (µg/m³).
# Baselines very loosely reflect typical annual-mean orderings — used only
# as anchors for the synthetic generator.
CITIES = [
    {"city": "Delhi",     "lat": 28.61, "lon": 77.21, "pm25_base": 110.0},
    {"city": "Mumbai",    "lat": 19.08, "lon": 72.88, "pm25_base":  55.0},
    {"city": "Bangalore", "lat": 12.97, "lon": 77.59, "pm25_base":  35.0},
    {"city": "Chennai",   "lat": 13.08, "lon": 80.27, "pm25_base":  40.0},
    {"city": "Kolkata",   "lat": 22.57, "lon": 88.36, "pm25_base":  80.0},
    {"city": "Hyderabad", "lat": 17.39, "lon": 78.49, "pm25_base":  45.0},
]

# Health-risk thresholds (3-class) derived from PM2.5 24-h exposure (µg/m³).
# Loosely aligned with WHO / NAAQS bucketing — for demo purposes, not clinical use.
RISK_LOW_MAX = 35.0      # Low:      PM2.5 <= 35
RISK_MOD_MAX = 75.0      # Moderate: 35 <  PM2.5 <= 75
                         # High:     PM2.5 >  75

RISK_LABELS = ["Low", "Moderate", "High"]

RANDOM_SEED = 7
