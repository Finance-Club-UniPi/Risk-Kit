import numpy as np
import pandas as pd
import yfinance as yf

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


if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "IWM"]  # S&P 500, Nasdaq-100, Russell 2000
    data = yf.download(tickers, start="2023-01-01", end="2025-03-17", progress=False)
    # MultiIndex columns: (Close, SPY), (Close, QQQ), ... or single ticker -> plain Close
    prices = data["Close"].copy() if len(tickers) > 1 else data["Close"].to_frame(tickers[0])
    prices = prices.dropna()

    result = markowitz_portofolio(prices)

    print("Markowitz portfolio (random weights):")
    print(f"  Expected return (annualized): {result['expected_return']:.4f}")
    print(f"  Variance (annualized):        {result['variance']:.6f}")
    print(f"  Volatility (annualized):      {result['volatility']:.4f}")
    print("  Weights:")
    for name, w in zip(prices.columns, result["weights"]):
        print(f"    {name}: {w:.4f}")