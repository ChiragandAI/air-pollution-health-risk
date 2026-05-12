# Learning Path — Air Pollution Health Risk Prediction

A study path to get from "I built this" to "I can defend every line of this in an interview."
Plan for ~90 minutes in one sitting. Read files in the order below.

---

## Step 0 — Set the frame (5 min)

Open the **resume bullet** for this project in your tailored resume. Read the 3 bullets out loud.

Everything you study from here is to make sure every word in those bullets is something you can defend. If a bullet says "SHAP-based interpretability" you must be able to explain what SHAP is. If it says "stratified cross-validation" you must know why stratified.

---

## Step 1 — `src/config.py` (5 min)

The trivial file. Just notice:
- The 6 cities with lat/lon and PM2.5 baselines.
- `RANDOM_SEED = 7` — that's why your results are reproducible.

**Lock in:** "Why a seed?" → reproducibility; same numbers every run, so the interviewer can verify metrics if they clone the repo.

---

## Step 2 — `src/data.py` (15 min, **most important file**)

Read the `generate()` function top to bottom. For each line, ask *why is this number here?*

| Line | What it's modelling |
|---|---|
| `winter_peak = _seasonal(doy, …, phase=15)` | Stubble burning + temperature inversions peak in Dec–Jan in North India |
| `weather_effect = -6.0 * wind_speed - 0.8 * rainfall` | Wind disperses, rain washes out |
| `weekend_dip = … dow >= 5` | Fewer vehicles on Sat/Sun |
| `pm10 = pm25 * 1.4-1.9` | Real-world ratio is typically 1.5–2× |

Then read the composite-label block at the bottom.

**Interview questions to be ready for:**

1. *"Why is the label a composite and not just PM2.5 thresholded?"*
   → If label = f(PM2.5) and feature = PM2.5, the model learns the threshold trivially. **That's target leakage.** A composite with noise gives something real to learn. Real-world health risk *is* multifactorial — PM2.5 dominates but PM10, NO₂, and humidity all contribute.

2. *"How would you swap this for real data?"*
   → Replace `generate()` with a CPCB API loader or an `openaq` Python client. The schema stays the same so `features.py` and `train.py` don't change. **That separation is the point** — data source is one swappable concern.

3. *"How do you know your simulator isn't lying?"*
   → I don't fully. It embeds known relationships (winter peak, wind reduces PM, weekend dip) so the rank-order of cities and the seasonal pattern match CPCB annual reports. Absolute values are anchored to typical baselines, not validated.

---

## Step 3 — `src/features.py` (15 min)

Three feature families. Understand the *motivation* for each.

### 3a. Rolling exposure windows (`pm25_roll3`, `pm25_roll7`)
- **Why:** Health effects of PM2.5 are about *cumulative* exposure, not single days. WHO's 24-hour guideline is one window; multi-day means matter for respiratory inflammation.
- **Mechanics:** `groupby('city').transform(rolling mean)` — `groupby` is critical, otherwise you'd roll across cities.
- **⚠️ Leakage trap:** A rolling mean that includes *today* leaks today's PM2.5 into the feature. We use `min_periods=1` and include today on purpose because the model uses today's PM2.5 as a separate column anyway. In a real forecasting setup (predicting tomorrow), you'd shift by 1.

### 3b. Lag features (`pm25_lag1`, `pm25_lag2`)
- **Why:** Yesterday's pollution predicts today's (autocorrelation in time series).
- **`.shift(lag).fillna(df['pm25'])`** — first-day fill avoids NaNs. Honest version: in strict time-series, you'd drop the first `max(lag)` rows per city.

### 3c. Cyclic day-of-year (`doy_sin`, `doy_cos`)
- **Why:** Day 365 and day 1 are one day apart in reality but 364 apart numerically. Encoding as `(sin(2πd/365), cos(2πd/365))` puts them next to each other on a circle.
- **Always asked:** *"Why not just use month as a categorical?"* → You could, but 12 one-hot columns lose the smooth gradient. Feb is closer to Jan than to July; one-hot says they're equidistant.

### 3d. City one-hot
- **Why one-hot and not label-encoded (Delhi=0, Mumbai=1)?** → Label encoding implies an ordering — "Mumbai is between Delhi and Bangalore" — which is meaningless. One-hot is right for nominal categories with low cardinality.

**Interview questions:**

1. *"What's the leakage risk in your feature pipeline?"*
   → Two: (a) rolling means that include "today" — fine here, problematic if forecasting. (b) lag features filled with today's value — same caveat.

2. *"Why 3-day and 7-day windows? Why not 1, 14, 30?"*
   → 3-day for short-term respiratory response, 7-day for weekly cumulative exposure. **Honest answer:** in a real project I'd treat windows as hyperparameters and tune via CV, or follow epidemiology literature (e.g., Schwartz et al. on PM lag structures).

---

## Step 4 — `src/train.py` (25 min, **second most important**)

Walk through each of the 4 functions: `train_classifier`, `train_regressor`, `cluster_zones`, `shap_summary`.

### 4a. Classifier — concepts to nail

- **RandomForest vs GradientBoosting:**
  - RF: many trees in parallel on bootstrapped data, averaged — *reduces variance*.
  - GB: trees sequentially, each fixing the previous one's residuals — *reduces bias*.
  - RF more robust to noise; GB usually wins on clean tabular but overfits faster.

- **StratifiedKFold vs KFold:**
  - "High" class is only ~12% of the data. Plain KFold could randomly give one fold zero "High" examples, breaking the metric.
  - Stratified keeps class proportions per fold.

- **f1_macro vs accuracy:**
  - Accuracy hides class imbalance — predicting "Low" for everything would score ~60%.
  - f1_macro is the unweighted mean of per-class F1 → forces good performance on rare classes.

- **Held-out 20% split + confusion matrix:** CV gives you a *score*; a confusion matrix gives you *what specific errors* the model makes. You need both.

### 4b. Regressor — concepts to nail

- **RMSE vs MAE vs R²:**
  - **RMSE** penalises big errors more (squared) — use when big errors are disproportionately bad.
  - **MAE** robust to outliers — use when all errors hurt equally.
  - **R²** is relative — fraction of variance explained. 0.978 is great, but on a synthetic dataset that's partly because AQI ≈ 1.6·PM2.5 + noise (quasi-linear target).

### 4c. Clustering — concepts to nail

- **Why standardize (`StandardScaler`) before KMeans?** → KMeans uses Euclidean distance; without standardization, the feature with the biggest numerical range dominates. PM2.5 (range 5–600) would swamp humidity (15–100).
- **Why k=3?** → Honest answer: hardcoded to match the 3 risk classes. In a real project, pick k via the **elbow method** (inertia vs k) or **silhouette score**.
- **KMeans assumptions:** spherical clusters of similar size. When that breaks, switch to DBSCAN (density-based) or hierarchical.

### 4d. SHAP — concepts to nail

- **What is a Shapley value?** Game theory: if N features cooperate to produce a prediction, the Shapley value is each player's fair share of the credit, averaged over all orderings.
- **TreeExplainer is exact** for tree ensembles (polynomial time, not exponential).
- **Why GB-multiclass failed:** sklearn's `GradientBoostingClassifier` for multi-class uses one-vs-rest internally; SHAP's TreeExplainer doesn't unpack that. So we tiebreak toward RF.

**Interview questions:**

1. *"Why stratified 5-fold and not time-series CV?"* — **see Weak Spots section below**.
2. *"Your model is 87% accurate; is that good?"* → Depends on baseline. Majority-class baseline = 60%. We're 27 points above. Macro-F1 0.84 tells me the gain is across all classes.
3. *"How would you handle class imbalance?"* → (a) `class_weight='balanced'`, (b) oversample High class (SMOTE), (c) threshold tuning on predicted probabilities. We didn't do any — macro-F1 was acceptable without it.

---

## Step 5 — `pipeline.py` (5 min)

The conductor. Dependency order: data → features → models → plots → save. `python3 pipeline.py` should always produce identical `outputs/` from a clean checkout.

**Interview question:** *"How is this reproducible?"* → seed + deterministic models (`random_state=7`) + pinned `requirements.txt`. Caveat: not bit-identical across OS/BLAS versions, but metrics within rounding.

---

## Step 6 — `app.py` (10 min)

Cache decorators:
- `@st.cache_data` — caches **return values** of pure functions (data loading). Re-runs if inputs change.
- `@st.cache_resource` — caches **objects you don't want to copy** (models, DB connections). One shared instance.

The "predict a custom day" form rebuilds the feature row by appending the user's input to the city's history and re-running `build_feature_table` — ensures rolling/lag features are computed exactly like in training.

**Interview question:** *"Why Streamlit and not FastAPI + React?"* → Streamlit is right when (a) users are scientists, not consumers, (b) ship in a day, (c) state is mostly session-local. FastAPI + JS wins when you need multi-tenant auth, persistent state, or 100+ concurrent users.

---

## Step 7 — Run the dashboard with `outputs/metrics.json` open (10 min)

`cat outputs/metrics.json` — memorise the actual numbers:
- **Best model:** RandomForest (tiebreak).
- **CV accuracy:** ~0.87, macro-F1 ~0.84.
- **AQI regression:** RMSE ~10, R² ~0.98.
- **Clusters:** Delhi → High; Kolkata → Moderate; rest → Low.

If asked "what numbers did you get?" you should recite these without looking.

---

## ⚠️ Weak Spots — know these *before* they ask

The candidate who volunteers these first looks senior; the one caught by them looks junior.

### 1. Stratified CV is *wrong* for time series — and I used it.
The right approach for daily environmental data is **`TimeSeriesSplit`** (train on days 1–200, test on 201–250; then 1–250 → 251–300; etc.). Stratified CV shuffles days randomly, so the model sees tomorrow's data while predicting today — temporal leakage.

**How to defend it:** *"I chose stratified because the 'High' class is sparse and TimeSeriesSplit on a single year would have given me folds with zero High examples. For a real deployment I'd use a grouped time-series split — block by month, stratify-within-block, or move to 3+ years of data so TimeSeriesSplit becomes viable."*

### 2. The data is synthetic.
Don't hide it. Lead with it: *"It's a deterministic simulator I wrote that bakes in known seasonal + weather + city effects. The architecture is data-source-agnostic — swap the loader for CPCB and nothing else changes."*

### 3. Geospatial is light.
Lat/lon are features, the dashboard plots them on a map, but there's no IDW interpolation, no kriging, no satellite-derived exposure layers. Next step: MODIS AOD or Sentinel-5P NO₂ rasters + IDW/kriging for unmonitored locations.

### 4. The label is engineered, not measured.
Real health-risk labels come from clinical outcomes (admissions, mortality, biomarkers). Mine is a composite proxy. Acknowledge it.

### 5. No hyperparameter tuning.
RF with `n_estimators=300` is a default. No GridSearchCV / Optuna. *"I deliberately kept this as a baseline to demonstrate the pipeline shape. Hyperparameter tuning is the next obvious improvement."*

### 6. 87% accuracy: how good is "good"?
Always have the baseline ready. Majority-class baseline = 60%. Random = 33%. So we're 27 points above majority.

---

## 90-minute schedule

| Time | What |
|---|---|
| 0:00–0:05 | Read the resume bullet again |
| 0:05–0:10 | `config.py` |
| 0:10–0:25 | `data.py` — domain logic + composite label |
| 0:25–0:40 | `features.py` — three feature families + leakage |
| 0:40–1:05 | `train.py` — classifier, regressor, clustering, SHAP |
| 1:05–1:10 | `pipeline.py` |
| 1:10–1:20 | `app.py` + run it, click around |
| 1:20–1:30 | Read the Weak Spots section twice; rehearse out loud |

If you can survive a friend asking "why did you do X?" for every X here, you're interview-proof on this project.
