import numpy as np


def _to_2d_array(array_1d_or_2d):
    """Convert input returns to a 2D NumPy array with basic validation."""
    returns = np.asarray(array_1d_or_2d, dtype=float)

    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)

    if returns.size == 0:
        raise ValueError("Input returns array is empty.")

    if returns.shape[0] < 2:
        raise ValueError("Need at least 2 time points.")

    if returns.shape[1] < 2:
        raise ValueError(
            "Single-asset portfolios are not supported. "
            "Portfolio optimization requires at least 2 assets."
        )

    return returns


def estimate_mean_cov(returns, annualization_factor=252):
    """Estimate annualized mean returns and covariance from raw returns."""
    returns = _to_2d_array(returns)

    expected_returns = np.mean(returns, axis=0) * annualization_factor
    cov = np.cov(returns, rowvar=False, ddof=1) * annualization_factor

    return expected_returns, cov


def portfolio_return(weights, expected_returns):
    """Compute portfolio expected return for given weights and asset returns."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    returns = np.asarray(expected_returns, dtype=float).reshape(-1)

    if weights.shape[0] != returns.shape[0]:
        raise ValueError("Dimension mismatch.")

    return float(weights @ returns)


def portfolio_volatility(weights, cov):
    """Compute portfolio volatility for given weights and covariance."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)

    if cov.ndim == 0:
        raise ValueError(
            "Covariance must be a 2D matrix, not a scalar. "
            "Single-asset portfolios are not supported."
        )

    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance must be square.")

    if weights.shape[0] != cov.shape[0]:
        raise ValueError("Dimension mismatch.")

    portfolio_variance = float(weights @ cov @ weights)
    return float(np.sqrt(max(portfolio_variance, 0.0)))


def global_minimum_variance_portfolio(cov):
    """Return fully invested weights of the global minimum-variance portfolio."""
    cov = np.asarray(cov, dtype=float)

    if cov.ndim == 0:
        raise ValueError(
            "Covariance must be a 2D matrix, not a scalar. "
            "Single-asset portfolios are not supported."
        )

    num_assets = cov.shape[0]

    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance must be square.")

    ones = np.ones(num_assets)
    inv_cov = np.linalg.pinv(cov)
    gmvp_weights = (inv_cov @ ones) / (ones @ inv_cov @ ones)

    return gmvp_weights


def efficient_frontier_weights(expected_returns, cov, target_return):
    """Markowitz efficient frontier weights for a given target return."""
    returns = np.asarray(expected_returns, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)

    if cov.ndim == 0:
        raise ValueError(
            "Covariance must be a 2D matrix, not a scalar. "
            "Single-asset portfolios are not supported."
        )

    num_assets = returns.shape[0]

    if cov.shape != (num_assets, num_assets):
        raise ValueError("Dimension mismatch.")

    ones = np.ones(num_assets)
    inv_cov = np.linalg.pinv(cov)

    a = float(ones @ inv_cov @ ones)
    b = float(ones @ inv_cov @ returns)
    c = float(returns @ inv_cov @ returns)
    determinant = a * c - b * b

    if abs(determinant) < 1e-14:
        raise ValueError("Degenerate system.")

    lam = (c - b * target_return) / determinant
    gam = (a * target_return - b) / determinant

    weights = inv_cov @ (
        lam * ones + gam * returns
    )

    return weights


def tangency_portfolio(expected_returns, cov, risk_free_rate=0.0):
    """Tangency (max Sharpe) portfolio relative to a risk-free rate."""
    returns = np.asarray(expected_returns, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)

    if cov.ndim == 0:
        raise ValueError(
            "Covariance must be a 2D matrix, not a scalar. "
            "Single-asset portfolios are not supported."
        )

    excess_returns = returns - risk_free_rate
    inv_cov = np.linalg.pinv(cov)
    raw_weights = inv_cov @ excess_returns
    weights_sum = float(np.sum(raw_weights))

    if abs(weights_sum) < 1e-14:
        raise ValueError("Cannot normalize weights.")
    return raw_weights / weights_sum


def efficient_frontier(expected_returns, cov, n_points=25):
    """Efficient frontier points over a grid of target returns."""
    returns = np.asarray(expected_returns, dtype=float).reshape(-1)
    target_returns = np.linspace(
        float(np.min(returns)),
        float(np.max(returns)),
        int(n_points),
    )

    returns_list = []
    volatility_list = []
    weights_list = []

    for target_return in target_returns:
        frontier_weights = efficient_frontier_weights(returns, cov, target_return)
        returns_list.append(portfolio_return(frontier_weights, returns))
        volatility_list.append(portfolio_volatility(frontier_weights, cov))
        weights_list.append(frontier_weights)

    return np.array(returns_list), np.array(volatility_list), weights_list


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.01, size=(1000, 3))

    expected_returns, cov = estimate_mean_cov(returns)
    weights = tangency_portfolio(expected_returns, cov, risk_free_rate=0.02)

    print(weights)
    print(portfolio_return(weights, expected_returns))
    print(portfolio_volatility(weights, cov))
