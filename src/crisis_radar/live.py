"""Live data prediction module for real-time risk assessment."""

import logging
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import CrisisRadarConfig
from .data import download_market_data
from .features import build_features, get_feature_names
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def get_live_risk_score(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    force_refresh: bool = True,
) -> Dict[str, any]:
    """
    Get live risk score using the latest market data.
    
    Args:
        model_path: Path to saved model. If None, uses default from artifacts.
        config_path: Path to saved config. If None, uses default from artifacts.
        force_refresh: If True, forces fresh data download (ignores cache).
    
    Returns:
        Dictionary with risk score, probability, date, and risk level.
    """
    # Load config
    if config_path is None:
        config_path = Path("artifacts/configs/pipeline_config.json")
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}. Run pipeline first.")
    
    from .utils import load_json
    config_dict = load_json(str(config_path))
    
    # Create config object
    config = CrisisRadarConfig(
        horizon_days=config_dict["horizon_days"],
        drawdown_threshold=config_dict["drawdown_threshold"],
        model_type=config_dict["model_type"],
        calibration_method=config_dict["calibration_method"],
        random_seed=config_dict["random_seed"],
    )
    
    # Load model
    if model_path is None:
        model_path = Path("artifacts/models/calibrated_model.pkl")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run pipeline first.")
    
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Download latest data (force refresh for live data)
    logger.info("Downloading latest market data...")
    today = date.today().strftime("%Y-%m-%d")
    
    # Get data from last 2 years to ensure we have enough history for features
    start_date = (date.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    
    cache_path = None if force_refresh else str(Path(config.data_dir) / "raw" / "market_data.csv")
    
    df_data = download_market_data(
        tickers=["^GSPC", "^VIX"],
        start=start_date,
        end=today,
        cache_path=cache_path,
    )
    
    df_spx = df_data["GSPC_Close"]
    df_vix = df_data["VIX_Close"]
    
    # Build features for latest date
    logger.info("Building features for latest date...")
    df_features = build_features(df_spx, df_vix)
    
    # Get the latest row (most recent trading day)
    if len(df_features) == 0:
        raise ValueError("No features available. Check data download.")
    
    latest_features = df_features.iloc[[-1]]
    latest_date = latest_features.index[0]
    
    # Prepare feature vector
    feature_names = get_feature_names()
    X_latest = latest_features[feature_names].values
    
    # Check for NaN values
    if np.isnan(X_latest).any():
        logger.warning("Some features are NaN. Using forward-filled values if available.")
        # Try to use previous day's values for NaN features
        if len(df_features) > 1:
            prev_features = df_features.iloc[[-2]][feature_names].values
            X_latest = np.where(np.isnan(X_latest), prev_features, X_latest)
    
    if np.isnan(X_latest).any():
        raise ValueError("Cannot compute prediction: some features are still NaN")
    
    # Predict probability
    logger.info(f"Computing risk score for {latest_date.date()}...")
    probability = model.predict_proba(X_latest)[0, 1]
    
    # Determine risk level
    if probability < 0.1:
        risk_level = "LOW"
    elif probability < 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    
    result = {
        "date": latest_date.strftime("%Y-%m-%d"),
        "probability": float(probability),
        "risk_level": risk_level,
        "horizon_days": config.horizon_days,
        "drawdown_threshold": config.drawdown_threshold,
    }
    
    logger.info(f"Risk Score: {probability:.2%} ({risk_level})")
    
    return result

