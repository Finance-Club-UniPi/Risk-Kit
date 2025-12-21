"""Plotting and visualization module."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def plot_probability_timeline(
    dates: pd.Index,
    probabilities: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "Crisis Probability Over Time",
) -> None:
    """
    Plot probability timeline with crash events highlighted.
    
    Args:
        dates: Date index
        probabilities: Predicted probabilities
        labels: True labels (1 = crash event)
        save_path: Path to save figure
        title: Plot title
    """
    logger.info(f"Creating probability timeline plot...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot probabilities
    ax.plot(dates, probabilities, color="blue", alpha=0.7, linewidth=1, label="Crisis Probability")
    
    # Highlight crash events
    crash_dates = dates[labels == 1]
    crash_probs = probabilities[labels == 1]
    
    if len(crash_dates) > 0:
        ax.scatter(
            crash_dates,
            crash_probs,
            color="red",
            s=50,
            alpha=0.8,
            zorder=5,
            label="Crash Events",
        )
        
        # Add vertical shading for crash events
        for crash_date in crash_dates:
            ax.axvline(x=crash_date, color="red", alpha=0.2, linestyle="--", linewidth=1)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Crisis Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved probability timeline to {save_path}")


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc: float,
    save_path: str,
    title: str = "Precision-Recall Curve",
) -> None:
    """
    Plot precision-recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        pr_auc: PR-AUC score
        save_path: Path to save figure
        title: Plot title
    """
    logger.info(f"Creating precision-recall curve...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color="blue", linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved precision-recall curve to {save_path}")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str,
    n_bins: int = 10,
    title: str = "Calibration Reliability Plot",
) -> None:
    """
    Plot calibration reliability curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save figure
        n_bins: Number of bins for calibration curve
        title: Plot title
    """
    logger.info(f"Creating calibration curve...")
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy="uniform"
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=1)
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", color="blue", label="Model", linewidth=2)
    
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved calibration curve to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
        title: Plot title
    """
    logger.info(f"Creating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )
    
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=["No Crash", "Crash"], yticklabels=["No Crash", "Crash"],
           title=title, ylabel="True Label", xlabel="Predicted Label")
    
    plt.tight_layout()
    
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_feature_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    save_path: str,
    title: str = "Feature Importance (Permutation)",
    n_repeats: int = 10,
) -> None:
    """
    Plot feature importance using permutation importance.
    
    Args:
        model: Fitted sklearn model
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        save_path: Path to save figure
        title: Plot title
        n_repeats: Number of times to permute each feature
    """
    logger.info(f"Computing feature importance...")
    
    # Compute permutation importance
    perm_importance = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )
    
    # Sort by mean importance
    indices = np.argsort(perm_importance.importances_mean)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(
        range(len(feature_names)),
        perm_importance.importances_mean[indices],
        xerr=perm_importance.importances_std[indices],
        color="steelblue",
    )
    
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved feature importance plot to {save_path}")

