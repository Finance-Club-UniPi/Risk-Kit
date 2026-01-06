import numpy as np
import pandas as pd

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily percentage returns from price data.
    """
    returns = prices.pct_change().dropna()
    return returns

def portofolio_expected_return(weights: np.ndarray, mean_returns: pd.Series) -> float:
    """
    Anualized expected portfolio return.
    """
    return np.sum(weights * mean_returns)*252

def portfolio_variance(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """
    Anualized portfolio variance.
    """
    return np.dot(weights.T, np.dot(cov_matrix*252, weights))

def markowitz_portofolio(prices: pd.DataFrame) -> dict:
    """
    Main Markowitz portofolio meltric.

    Returns:
    - expected return
    - variance
    - volatility
    - weights
    """

    returns = compute_returns(prices)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_assets = len(mean_returns)
    weights = np.random.random(num_assets)
    weights = weights / np.sum(weights)

    exp_return = portofolio_expected_return(weights, mean_returns)
    var = portfolio_variance(weights, cov_matrix)
    volatility = np.sqrt(var)

    return {
        "expected_return": exp_return,
        "variance": var,
        "volatility": volatility,
        "weights": weights
    }