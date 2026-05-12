# Air Pollution Health Risk Prediction

An end-to-end Python ML pipeline that **links air-quality, weather, and location data
to predict pollution-driven health-risk categories** and surface high-risk zones — a
small-scale demonstration of the kind of integration needed for environmental-health
research platforms (e.g. AIRCARE-style studies).

## What it does

1. **Data integration** — loads a multi-city air-quality + weather + location dataset
   (synthetic, reproducible, committed to the repo) and joins the sources into a
   single modelling table.
2. **Feature engineering** — rolling exposure windows (3-day, 7-day PM2.5 means),
   lag features, cyclic day-of-year encoding, and one-hot location encoding.
3. **Modelling** — three heads on the same feature table:
   - Health-risk **classification** (Low / Moderate / High) with Random Forest and
     Gradient Boosting, stratified 5-fold cross-validation.
   - AQI **regression** with a Random Forest regressor.
   - **KMeans clustering** over city-day aggregates to identify high-risk zones.
4. **Interpretability** — SHAP TreeExplainer values + feature-importance plots.
5. **Dashboard** — a Streamlit app showing per-city exposure trends, a risk map,
   feature importances, and on-the-fly predictions.

## Quickstart

```bash
# 1. Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the full pipeline end-to-end
#    (data -> features -> train -> evaluate -> outputs/)
python3 pipeline.py

# 3. Launch the dashboard
streamlit run app.py
```

Pipeline outputs land in `outputs/`:
- `metrics.json` — CV scores for classification, regression RMSE, cluster sizes.
- `feature_importance.png` — top features for the classifier.
- `shap_summary.png` — SHAP summary plot.
- `confusion_matrix.png` — classifier confusion matrix.
- `models/` — pickled trained models for the dashboard.

## Project layout

```
air-pollution-health-risk/
├── README.md
├── requirements.txt
├── pipeline.py             ← end-to-end orchestrator
├── app.py                  ← Streamlit dashboard
├── src/
│   ├── data.py             ← synthetic dataset generator + loader
│   ├── features.py         ← rolling / lag / cyclic / one-hot features
│   ├── train.py            ← classification, regression, clustering, SHAP
│   └── config.py           ← paths, constants, risk thresholds
├── data/
│   └── synthetic/
│       └── aircare_synthetic.csv   ← regenerated on first `pipeline.py` run
└── outputs/                ← created at runtime
```

## Honest scope

- **Data is synthetic, not real.** It's generated from realistic-looking distributions
  (city-specific PM2.5 baselines, seasonal patterns, weather-pollution correlations,
  weekend effects, noise) so the pipeline can run anywhere without scraping CPCB /
  OpenAQ. The generator lives in [src/data.py](src/data.py) — swap it for a real
  loader to plug in actual data.
- **Models are baselines.** Random Forest and Gradient Boosting are deliberately
  simple defaults; the point is to demonstrate the integration / feature-engineering
  / evaluation pattern, not to chase SOTA accuracy.
- **Geospatial is light.** Cities have lat/lon and the map uses them, but there's no
  raster / IDW / kriging. That would be the next step for a real environmental-health
  build.

## Why this project exists

It was built to plug a domain gap on Chirag Dahiya's resume when applying for an
environmental-health data-science role. The resume bullet pointing here is in the
repository at `01_2026-05-12_Data_Scientist_AIIMS/tailored_resume.html`.
