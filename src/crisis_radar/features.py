"""Feature engineering module."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_features(df_spx: pd.Series, df_vix: pd.Series) -> pd.DataFrame:
    """
    Build features from S&P 500 and VIX data.
    
    All features use only past information (no look-ahead bias).
    
    Args:
        df_spx: Series of S&P 500 Close prices, indexed by date
        df_vix: Series of VIX Close prices, indexed by date
    
    Returns:
        DataFrame with features, indexed by date
    """
    logger.info("Building features...")
    
    # Align indices
    common_idx = df_spx.index.intersection(df_vix.index)
    df_spx = df_spx.loc[common_idx]
    df_vix = df_vix.loc[common_idx]
    
    features = pd.DataFrame(index=common_idx)
    
    # Price-based features (using SPX)
    features["log_return_1d"] = np.log(df_spx / df_spx.shift(1))
    features["return_5d"] = (df_spx / df_spx.shift(5) - 1)
    features["return_20d"] = (df_spx / df_spx.shift(20) - 1)
    
    # Realized volatility (rolling std of returns, annualized)
    returns = df_spx.pct_change()
    features["realized_vol_20d"] = returns.rolling(window=20).std() * np.sqrt(252)
    
    # Momentum indicators
    sma_50 = df_spx.rolling(window=50).mean()
    sma_200 = df_spx.rolling(window=200).mean()
    features["momentum_50_200"] = (sma_50 / sma_200 - 1)
    
    # Drawdown from recent high
    rolling_max_60d = df_spx.rolling(window=60).max()
    features["drawdown_60d"] = (df_spx / rolling_max_60d - 1)
    
    # VIX features
    features["vix_level"] = df_vix
    features["vix_change_5d"] = (df_vix / df_vix.shift(5) - 1)
    features["vix_change_20d"] = (df_vix / df_vix.shift(20) - 1)
    
    # Optional: rolling skewness and kurtosis (60-day window)
    features["rolling_skew_60d"] = returns.rolling(window=60).skew()
    # Kurtosis: use apply with scipy or compute manually
    from scipy.stats import kurtosis
    features["rolling_kurtosis_60d"] = returns.rolling(window=60).apply(
        lambda x: kurtosis(x, nan_policy="omit") if len(x.dropna()) > 0 else np.nan,
        raw=False
    )
    
    # Drop rows with NaN (from rolling windows)
    initial_len = len(features)
    features = features.dropna()
    logger.info(f"Dropped {initial_len - len(features)} rows with NaN from feature engineering")
    
    logger.info(f"Built {len(features.columns)} features for {len(features)} dates")
    return features


def get_feature_names() -> list:
    """Return list of feature names."""
    return [
        "log_return_1d",
        "return_5d",
        "return_20d",
        "realized_vol_20d",
        "momentum_50_200",
        "drawdown_60d",
        "vix_level",
        "vix_change_5d",
        "vix_change_20d",
        "rolling_skew_60d",
        "rolling_kurtosis_60d",
    ]

