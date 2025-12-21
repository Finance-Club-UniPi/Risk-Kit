"""Model evaluation and metrics computation module."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        threshold: Decision threshold. If None, uses threshold that maximizes F1.
    
    Returns:
        Dictionary of metrics
    """
    # Compute PR curve to find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Find threshold that maximizes F1
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
    
    if threshold is None:
        threshold = optimal_threshold
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Compute metrics
    metrics = {
        "pr_auc": average_precision_score(y_true, y_pred_proba),  # PR-AUC
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "brier_score": brier_score_loss(y_true, y_pred_proba),
        "threshold": float(threshold),
        "optimal_threshold": float(optimal_threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    
    return metrics, precision, recall, thresholds


def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model and save metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Optional path to save metrics JSON
    
    Returns:
        Tuple of (metrics_dict, precision, recall, thresholds)
    """
    logger.info("Computing evaluation metrics...")
    
    metrics, precision, recall, thresholds = compute_metrics(y_true, y_pred_proba)
    
    logger.info(f"PR-AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Brier Score: {metrics['brier_score']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1: {metrics['f1']:.4f}")
    logger.info(f"Optimal threshold: {metrics['optimal_threshold']:.4f}")
    
    if save_path:
        ensure_dir(Path(save_path).parent)
        save_json(metrics, save_path)
        logger.info(f"Saved metrics to {save_path}")
    
    return metrics, precision, recall, thresholds

