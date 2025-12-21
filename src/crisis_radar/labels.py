"""Target label creation module."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_future_drawdown_label(
    close_series: pd.Series,
    horizon_days: int,
    drawdown_threshold: float,
) -> pd.Series:
    """
    Create binary label for future drawdown events.
    
    Label[t] = 1 if the maximum drawdown from close[t] within the next
    horizon_days trading days is <= -drawdown_threshold.
    
    Args:
        close_series: Series of closing prices, indexed by date
        horizon_days: Number of trading days to look ahead (N)
        drawdown_threshold: Drawdown threshold as positive fraction (D, e.g., 0.08 for -8%)
    
    Returns:
        Series of binary labels {0, 1}, indexed by date
    """
    logger.info(f"Creating labels: horizon={horizon_days} days, threshold={drawdown_threshold}")
    
    labels = pd.Series(index=close_series.index, dtype=int)
    labels[:] = 0
    
    # For each date, look ahead up to horizon_days
    for i in range(len(close_series)):
        current_price = close_series.iloc[i]
        current_date = close_series.index[i]
        
        # Find the end index (up to horizon_days ahead)
        end_idx = min(i + horizon_days, len(close_series))
        
        if end_idx <= i + 1:
            # Not enough future data
            labels.iloc[i] = 0
            continue
        
        # Get future prices
        future_prices = close_series.iloc[i + 1 : end_idx]
        
        if len(future_prices) == 0:
            labels.iloc[i] = 0
            continue
        
        # Calculate minimum price in future window
        future_min = future_prices.min()
        
        # Calculate future drawdown
        future_drawdown = (future_min / current_price) - 1
        
        # Label = 1 if drawdown <= -threshold
        if future_drawdown <= -drawdown_threshold:
            labels.iloc[i] = 1
    
    # Set label to NaN for the last horizon_days rows (can't compute future drawdown)
    labels.iloc[-horizon_days:] = np.nan
    
    n_events = labels.sum()
    n_valid = labels.notna().sum()
    event_rate = n_events / n_valid if n_valid > 0 else 0
    
    logger.info(
        f"Label creation complete: {n_events} events ({event_rate:.2%}) "
        f"out of {n_valid} valid labels"
    )
    
    return labels

