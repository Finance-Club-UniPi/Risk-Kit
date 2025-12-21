"""Time-series data splitting module."""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def make_time_split(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and test sets based on date ranges.
    
    Args:
        df: DataFrame indexed by date
        train_start: Start date for training set (YYYY-MM-DD)
        train_end: End date for training set (YYYY-MM-DD)
        test_start: Start date for test set (YYYY-MM-DD)
        test_end: End date for test set (YYYY-MM-DD)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Splitting data: train={train_start} to {train_end}, test={test_start} to {test_end}")
    
    train_df = df.loc[train_start:train_end].copy()
    test_df = df.loc[test_start:test_end].copy()
    
    logger.info(f"Train set: {len(train_df)} rows")
    logger.info(f"Test set: {len(test_df)} rows")
    
    if len(train_df) == 0:
        raise ValueError("Training set is empty")
    if len(test_df) == 0:
        raise ValueError("Test set is empty")
    
    return train_df, test_df

