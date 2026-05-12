"""End-to-end orchestrator: data → features → train → evaluate → outputs/.

Run:
    python3 pipeline.py
"""

from __future__ import annotations

import json

from src import config, data, features, train


def main() -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/5  Loading / generating dataset …")
    df = data.load_or_generate()
    print(f"     rows={len(df)}  cities={df['city'].nunique()}  "
          f"dates={df['date'].min().date()} → {df['date'].max().date()}")
    print(f"     risk-label balance: "
          f"{df['risk_label'].value_counts().to_dict()}")

    print("\n2/5  Building feature table …")
    X, y_class, y_reg, _meta = features.build_feature_table(df)
    print(f"     X.shape={X.shape}  n_features={X.shape[1]}")

    print("\n3/5  Classification (5-fold CV; RF vs GB) …")
    clf, clf_metrics = train.train_classifier(X, y_class)

    print("\n4/5  Regression on AQI …")
    reg, reg_metrics = train.train_regressor(X, y_reg)

    print("\n5/5  Clustering — high-risk zone discovery …")
    kmeans, cluster_metrics = train.cluster_zones(df)

    # Plots & artifacts
    train.plot_feature_importance(clf, X.columns.tolist(),
                                  config.OUTPUTS_DIR / "feature_importance.png")
    train.plot_confusion(clf_metrics["confusion_matrix"],
                         clf_metrics["labels"],
                         config.OUTPUTS_DIR / "confusion_matrix.png")
    train.shap_summary(clf, X, config.OUTPUTS_DIR / "shap_summary.png")
    train.save_models(clf, reg, kmeans)

    result = train.TrainResult(
        classification=clf_metrics,
        regression=reg_metrics,
        clustering=cluster_metrics,
    )
    metrics_path = config.OUTPUTS_DIR / "metrics.json"
    metrics_path.write_text(result.to_json())
    print(f"\nWrote: {metrics_path}")
    print(f"Wrote: {config.OUTPUTS_DIR / 'feature_importance.png'}")
    print(f"Wrote: {config.OUTPUTS_DIR / 'confusion_matrix.png'}")
    print(f"Wrote: {config.OUTPUTS_DIR / 'shap_summary.png'}")
    print(f"Wrote: {config.MODELS_DIR}/classifier.joblib, regressor.joblib, kmeans.joblib")
    print("\n✓ Pipeline complete. Run `streamlit run app.py` for the dashboard.")


if __name__ == "__main__":
    main()
