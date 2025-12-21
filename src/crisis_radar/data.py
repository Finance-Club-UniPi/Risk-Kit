"""Data download and caching module."""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def download_market_data(
    tickers: List[str],
    start: str,
    end: str,
    cache_path: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download market data using yfinance and optionally cache it.
    
    Args:
        tickers: List of ticker symbols (e.g., ["^GSPC", "^VIX"])
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD)
        cache_path: Optional path to cache file. If file exists, load from cache.
        force_refresh: If True, ignore cache and download fresh data.
    
    Returns:
        DataFrame with columns for each ticker's Close price, indexed by date.
        Columns are named like "SPX_Close", "VIX_Close" based on ticker symbols.
    """
    if cache_path and Path(cache_path).exists() and not force_refresh:
        logger.info(f"Loading cached data from {cache_path}")
        # Read without parse_dates first to handle timezone issues
        df = pd.read_csv(cache_path, index_col=0)
        # Convert index to timezone-naive DatetimeIndex
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        # Filter to requested date range (timezone-naive Timestamp)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        # Use boolean indexing for filtering
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        df = df.loc[mask]
        return df
    
    logger.info(f"Downloading data for {tickers} from {start} to {end}")
    
    all_data = {}
    for ticker in tickers:
        try:
            logger.info(f"Downloading {ticker}...")
            ticker_obj = yf.Ticker(ticker)
            df_ticker = ticker_obj.history(start=start, end=end)
            
            if df_ticker.empty:
                logger.warning(f"No data returned for {ticker}")
                continue
            
            # Use Close price and rename column
            ticker_name = ticker.replace("^", "").replace("-", "_")
            all_data[f"{ticker_name}_Close"] = df_ticker["Close"]
            
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            raise
    
    if not all_data:
        raise ValueError("No data downloaded for any ticker")
    
    # Combine into single DataFrame
    df = pd.DataFrame(all_data)
    df.index.name = "Date"
    
    # Align all series to trading calendar (forward-fill VIX on holidays)
    # Only forward-fill if there are missing values
    df = df.sort_index()
    
    # Forward-fill missing values (e.g., VIX on holidays)
    # But only for a limited number of days to avoid stale data
    df = df.ffill(limit=5)
    
    # Drop any remaining rows with NaN (should be minimal)
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with NaN values")
    
    # Cache if path provided
    if cache_path:
        ensure_dir(Path(cache_path).parent)
        df.to_csv(cache_path)
        logger.info(f"Cached data to {cache_path}")
    
    logger.info(f"Downloaded {len(df)} rows of data")
    return df

