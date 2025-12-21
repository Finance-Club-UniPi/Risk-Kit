"""End-to-end pipeline orchestration module."""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .calibration import calibrate_model
from .config import CrisisRadarConfig
from .data import download_market_data
from .evaluation import evaluate_model
from .features import build_features, get_feature_names
from .labels import create_future_drawdown_label
from .models import get_models
from .plotting import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_probability_timeline,
)
from .split import make_time_split
from .utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


def run_pipeline(config: CrisisRadarConfig) -> Dict[str, str]:
    """
    Run the complete Crisis Radar ML pipeline.
    
    Args:
        config: CrisisRadarConfig instance
    
    Returns:
        Dictionary with paths to saved artifacts
    """
    logger.info("=" * 60)
    logger.info("Starting Crisis Radar Pipeline")
    logger.info("=" * 60)
    
    artifacts = {}
    
    # Step 1: Download data
    logger.info("\n[1/8] Downloading market data...")
    cache_path = Path(config.data_dir) / "raw" / "market_data.csv"
    df_data = download_market_data(
        tickers=["^GSPC", "^VIX"],
        start=config.start_date,
        end=config.end_date,
        cache_path=str(cache_path),
        force_refresh=False,  # Use cache for pipeline runs
    )
    
    df_spx = df_data["GSPC_Close"]
    df_vix = df_data["VIX_Close"]
    
    # Step 2: Build features
    logger.info("\n[2/8] Building features...")
    df_features = build_features(df_spx, df_vix)
    
    # Step 3: Create labels
    logger.info("\n[3/8] Creating labels...")
    labels = create_future_drawdown_label(
        df_spx,
        horizon_days=config.horizon_days,
        drawdown_threshold=config.drawdown_threshold,
    )
    
    # Combine features and labels
    df_full = df_features.copy()
    df_full["label"] = labels
    
    # Drop rows with NaN labels (end of series)
    df_full = df_full.dropna(subset=["label"])
    
    # Step 4: Split data
    logger.info("\n[4/8] Splitting data...")
    train_df, test_df = make_time_split(
        df_full,
        train_start=config.train_start,
        train_end=config.train_end,
        test_start=config.test_start,
        test_end=config.test_end,
    )
    
    # Prepare feature matrix and labels
    feature_names = get_feature_names()
    X_train = train_df[feature_names].values
    y_train = train_df["label"].values.astype(int)
    X_test = test_df[feature_names].values
    y_test = test_df["label"].values.astype(int)
    
    logger.info(f"Train: {len(X_train)} samples, {y_train.sum()} events ({y_train.mean():.2%})")
    logger.info(f"Test: {len(X_test)} samples, {y_test.sum()} events ({y_test.mean():.2%})")
    
    # Step 5: Train model
    logger.info("\n[5/8] Training model...")
    models = get_models(config)
    model = models[config.model_type]
    
    logger.info(f"Training {config.model_type} model...")
    model.fit(X_train, y_train)
    
    # Step 6: Calibrate probabilities
    logger.info("\n[6/8] Calibrating probabilities...")
    calibrated_model = calibrate_model(
        model,
        X_train,
        y_train,
        method=config.calibration_method,
    )
    
    # Step 7: Evaluate on test set
    logger.info("\n[7/8] Evaluating on test set...")
    y_test_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    metrics_path = Path(config.artifacts_dir) / "metrics" / "test_metrics.json"
    metrics, precision, recall, thresholds = evaluate_model(
        y_test,
        y_test_proba,
        save_path=str(metrics_path),
    )
    artifacts["metrics"] = str(metrics_path)
    
    # Step 8: Generate plots
    logger.info("\n[8/8] Generating plots...")
    figures_dir = Path(config.reports_dir) / "figures"
    
    # Probability timeline
    plot_probability_timeline(
        test_df.index,
        y_test_proba,
        y_test,
        save_path=str(figures_dir / "probability_timeline.png"),
    )
    artifacts["probability_timeline"] = str(figures_dir / "probability_timeline.png")
    
    # Precision-Recall curve
    plot_precision_recall_curve(
        precision,
        recall,
        metrics["pr_auc"],
        save_path=str(figures_dir / "precision_recall_curve.png"),
    )
    artifacts["pr_curve"] = str(figures_dir / "precision_recall_curve.png")
    
    # Calibration curve
    plot_calibration_curve(
        y_test,
        y_test_proba,
        save_path=str(figures_dir / "calibration_curve.png"),
    )
    artifacts["calibration_curve"] = str(figures_dir / "calibration_curve.png")
    
    # Confusion matrix
    threshold = metrics["optimal_threshold"]
    y_test_pred = (y_test_proba >= threshold).astype(int)
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        save_path=str(figures_dir / "confusion_matrix.png"),
    )
    artifacts["confusion_matrix"] = str(figures_dir / "confusion_matrix.png")
    
    # Feature importance
    plot_feature_importance(
        calibrated_model,
        X_test,
        y_test,
        feature_names,
        save_path=str(figures_dir / "feature_importance.png"),
    )
    artifacts["feature_importance"] = str(figures_dir / "feature_importance.png")
    
    # Save model and artifacts
    logger.info("\nSaving model artifacts...")
    model_path = Path(config.artifacts_dir) / "models" / "calibrated_model.pkl"
    ensure_dir(model_path.parent)
    with open(model_path, "wb") as f:
        pickle.dump(calibrated_model, f)
    artifacts["model"] = str(model_path)
    
    # Save scaler (if applicable)
    if hasattr(model, "named_steps") and "scaler" in model.named_steps:
        scaler_path = Path(config.artifacts_dir) / "models" / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(model.named_steps["scaler"], f)
        artifacts["scaler"] = str(scaler_path)
    
    # Save config
    config_path = Path(config.artifacts_dir) / "configs" / "pipeline_config.json"
    config_dict = {
        "horizon_days": config.horizon_days,
        "drawdown_threshold": config.drawdown_threshold,
        "model_type": config.model_type,
        "calibration_method": config.calibration_method,
        "train_start": config.train_start,
        "train_end": config.train_end,
        "test_start": config.test_start,
        "test_end": config.test_end,
        "random_seed": config.random_seed,
    }
    save_json(config_dict, str(config_path))
    artifacts["config"] = str(config_path)
    
    # Save predictions for later analysis
    predictions_path = Path(config.artifacts_dir) / "metrics" / "test_predictions.csv"
    pred_df = pd.DataFrame({
        "date": test_df.index,
        "true_label": y_test,
        "predicted_probability": y_test_proba,
        "predicted_label": y_test_pred,
    })
    pred_df.to_csv(predictions_path, index=False)
    artifacts["predictions"] = str(predictions_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"\nArtifacts saved to: {config.artifacts_dir}")
    logger.info(f"Figures saved to: {config.reports_dir}/figures")
    logger.info(f"\nTest Set Metrics:")
    logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1: {metrics['f1']:.4f}")
    
    return artifacts

