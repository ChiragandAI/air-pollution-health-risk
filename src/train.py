"""Training, evaluation, clustering, and interpretability for the pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class TrainResult:
    classification: dict[str, Any] = field(default_factory=dict)
    regression: dict[str, Any] = field(default_factory=dict)
    clustering: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ───────────────────────── Classification ─────────────────────────

def train_classifier(X: pd.DataFrame, y: pd.Series) -> tuple[Any, dict[str, Any]]:
    """Train RF + GB classifiers with 5-fold stratified CV; keep the better one."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    y_codes = y.cat.codes if hasattr(y, "cat") else y

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=16,
            random_state=config.RANDOM_SEED, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, random_state=config.RANDOM_SEED
        ),
    }
    cv_scores: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        acc = cross_val_score(model, X, y_codes, cv=cv, scoring="accuracy", n_jobs=-1)
        f1m = cross_val_score(model, X, y_codes, cv=cv, scoring="f1_macro", n_jobs=-1)
        cv_scores[name] = {
            "accuracy_mean": float(acc.mean()),
            "accuracy_std": float(acc.std()),
            "f1_macro_mean": float(f1m.mean()),
            "f1_macro_std": float(f1m.std()),
        }
        print(f"  [{name}] acc={acc.mean():.3f}±{acc.std():.3f} "
              f"f1_macro={f1m.mean():.3f}±{f1m.std():.3f}")

    # Pick the better F1 model, fit on a held-out split for the confusion matrix.
    # Tiebreak (within 0.005 macro-F1) toward RandomForest — multi-class SHAP
    # supports tree ensembles but not multi-class GradientBoosting.
    ranked = sorted(cv_scores.items(),
                    key=lambda kv: kv[1]["f1_macro_mean"], reverse=True)
    best_name = ranked[0][0]
    if len(ranked) >= 2 and abs(ranked[0][1]["f1_macro_mean"]
                                - ranked[1][1]["f1_macro_mean"]) < 0.005:
        best_name = "RandomForest" if "RandomForest" in dict(ranked) else best_name
    best = models[best_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_codes, test_size=0.2, stratify=y_codes, random_state=config.RANDOM_SEED
    )
    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=config.RISK_LABELS,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    holdout_f1 = float(f1_score(y_test, y_pred, average="macro"))

    return best, {
        "cv_scores": cv_scores,
        "best_model": best_name,
        "holdout_f1_macro": holdout_f1,
        "holdout_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": config.RISK_LABELS,
    }


# ───────────────────────── Regression ─────────────────────────

def train_regressor(X: pd.DataFrame, y: pd.Series) -> tuple[Any, dict[str, Any]]:
    """Train an AQI regressor with a held-out split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED
    )
    model = RandomForestRegressor(
        n_estimators=120, max_depth=14,
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    print(f"  [Regressor] RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}")
    return model, {"rmse": rmse, "mae": mae, "r2": r2}


# ───────────────────────── Clustering ─────────────────────────

def cluster_zones(df: pd.DataFrame) -> tuple[KMeans, dict[str, Any]]:
    """KMeans over city-level mean exposure to discover high-risk zones."""
    agg = (
        df.groupby("city")
          .agg(pm25_mean=("pm25", "mean"),
               pm25_max=("pm25", "max"),
               pm10_mean=("pm10", "mean"),
               no2_mean=("no2", "mean"),
               aqi_mean=("aqi", "mean"))
          .reset_index()
    )
    feats = agg.drop(columns=["city"])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats)
    k = 3
    km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
    labels = km.fit_predict(scaled)
    agg["cluster"] = labels

    # Rank clusters by mean PM2.5 so the highest-PM cluster gets the highest id.
    order = agg.groupby("cluster")["pm25_mean"].mean().sort_values().index.tolist()
    rank_map = {c: rank for rank, c in enumerate(order)}
    agg["zone_risk_rank"] = agg["cluster"].map(rank_map)
    zone_names = {0: "Low-risk zone", 1: "Moderate-risk zone", 2: "High-risk zone"}
    agg["zone_name"] = agg["zone_risk_rank"].map(zone_names)

    print(f"  [Clusters] k={k}  zones:")
    for _, row in agg.sort_values("zone_risk_rank").iterrows():
        print(f"    - {row['city']:<10s} → {row['zone_name']:<18s} "
              f"(pm25_mean={row['pm25_mean']:.1f})")

    return km, {
        "k": k,
        "cities_by_zone": (agg.sort_values("zone_risk_rank")
                              [["city", "zone_name", "pm25_mean", "aqi_mean"]]
                              .to_dict(orient="records")),
    }


# ───────────────────────── Interpretability ─────────────────────────

def shap_summary(model, X: pd.DataFrame, out_path) -> None:
    """SHAP summary plot for a tree model. Sub-samples for runtime."""
    sample = X.sample(min(800, len(X)), random_state=config.RANDOM_SEED)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    plt.figure(figsize=(8, 6))
    # Multi-class trees → shap_values is a list per class; aggregate to mean |shap|.
    if isinstance(shap_values, list):
        agg = np.mean([np.abs(v) for v in shap_values], axis=0)
        shap.summary_plot(agg, sample, show=False, plot_type="bar")
    else:
        shap.summary_plot(shap_values, sample, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_feature_importance(model, feature_names, out_path) -> None:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return
    s = pd.Series(imp, index=feature_names).sort_values(ascending=True).tail(15)
    plt.figure(figsize=(7, 5))
    s.plot(kind="barh", color="#1e4d94")
    plt.title("Top features — classifier")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_confusion(cm, labels, out_path) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix (held-out 20%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def save_models(clf, reg, kmeans) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, config.MODELS_DIR / "classifier.joblib", compress=3)
    joblib.dump(reg, config.MODELS_DIR / "regressor.joblib", compress=3)
    joblib.dump(kmeans, config.MODELS_DIR / "kmeans.joblib", compress=3)
