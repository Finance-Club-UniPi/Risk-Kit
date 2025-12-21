"""Probability calibration module."""

import logging
from typing import Optional

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import PredefinedSplit

logger = logging.getLogger(__name__)


def calibrate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "isotonic",
    cv_folds: Optional[int] = None,
) -> CalibratedClassifierCV:
    """
    Calibrate a model's probability predictions.
    
    Uses a single validation split (last 20% of training data) to avoid issues
    with CV folds that may have only one class. Clones the model to avoid
    modifying the original fitted model.
    
    Args:
        model: Already fitted sklearn model
        X_train: Training features
        y_train: Training labels
        method: Calibration method ("isotonic" or "sigmoid")
        cv_folds: Ignored (kept for compatibility). Uses single validation split.
    
    Returns:
        CalibratedClassifierCV fitted model
    """
    logger.info(f"Calibrating model with method={method}")
    
    # Check if both classes are present in training data
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        logger.warning(
            f"Only {len(unique_classes)} class(es) present in training data. "
            "Skipping calibration and returning original model."
        )
        return model
    
    # Clone the model to avoid modifying the original
    model_clone = clone(model)
    
    # Use a single validation split (last 20% of training data) for calibration
    # This ensures we have both classes in the validation set if they exist in training
    split_idx = int(len(X_train) * 0.8)
    
    # Create a PredefinedSplit that uses the last 20% as validation
    # -1 means train, 0 means validation
    test_fold = np.full(len(X_train), -1, dtype=int)
    test_fold[split_idx:] = 0
    ps = PredefinedSplit(test_fold)
    
    # Verify validation set has both classes
    y_val = y_train[split_idx:]
    val_classes = np.unique(y_val)
    if len(val_classes) < 2:
        logger.warning(
            "Validation split has only one class. "
            "Skipping calibration and returning original model."
        )
        return model
    
    # Use PredefinedSplit for single validation fold
    calibrated = CalibratedClassifierCV(
        model_clone,
        method=method,
        cv=ps,
    )
    
    try:
        calibrated.fit(X_train, y_train)
    except ValueError as e:
        if "two classes" in str(e).lower():
            logger.warning(
                f"Calibration failed due to class imbalance: {e}. "
                "Skipping calibration and returning original model."
            )
            return model
        else:
            raise
    
    logger.info("Calibration complete")
    return calibrated

