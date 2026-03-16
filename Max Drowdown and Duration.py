import numpy as np


def _as_1d_float_array(portfolio_returns):
    """
    Convert input returns to a 1D numpy array of floats.
    """
    returns_array = np.asarray(portfolio_returns, dtype=float).reshape(-1)

    if returns_array.size == 0:
        raise ValueError("Input portfolio returns are empty.")

    return returns_array


def _validate_alpha(alpha):
    """
    Validate that alpha is between 0 and 1 (exclusive).
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")


def _round_float(value, decimals=4):
    """
    Round a float to a given number of decimal places.
    """
    return float(np.round(value, decimals))


def value_at_risk_historical(returns_array, alpha=0.05, decimals=4):
    """
    Historical Value at Risk (VaR).

    Parameters
    ----------
    returns_array : np.ndarray
        1D array of portfolio returns in decimal form.
    alpha : float
        Tail probability (e.g. 0.05 for 5% VaR).
    decimals : int
        Number of decimal places for the result.

    Returns
    -------
    float
        VaR as a positive loss number (rounded).
    """
    _validate_alpha(alpha)

    var_threshold_return = np.quantile(returns_array, alpha)
    var_loss = -var_threshold_return

    return _round_float(var_loss, decimals=decimals)


def cvar_expected_shortfall_historical(returns_array, alpha=0.05, decimals=4):
    """
    Historical Conditional VaR (CVaR) / Expected Shortfall.

    Parameters
    ----------
    returns_array : np.ndarray
        1D array of portfolio returns in decimal form.
    alpha : float
        Tail probability (e.g. 0.05 for worst 5%).
    decimals : int
        Number of decimal places for the result.

    Returns
    -------
    float
        CVaR as a positive loss number (rounded).
    """
    _validate_alpha(alpha)

    var_threshold_return = np.quantile(returns_array, alpha)
    tail_returns = returns_array[returns_array <= var_threshold_return]

    if tail_returns.size == 0:
        return float("nan")

    expected_shortfall_loss = np.mean(-tail_returns)

    return _round_float(expected_shortfall_loss, decimals=decimals)


def var_cvar_report(portfolio_returns, alpha_levels=(0.01, 0.05), decimals=4):
    """
    Compute VaR and CVaR for multiple alpha levels.

    Parameters
    ----------
    portfolio_returns : array-like
        Portfolio returns in decimal form.
    alpha_levels : iterable of float
        Tail probabilities (e.g. (0.01, 0.05)).
    decimals : int
        Number of decimal places for all results.

    Returns
    -------
    dict
        {alpha: {"VaR": value, "CVaR": value}, ...}
    """
    returns_array = _as_1d_float_array(portfolio_returns)

    report = {}

    for alpha in alpha_levels:
        alpha = float(alpha)

        var_value = value_at_risk_historical(
            returns_array,
            alpha=alpha,
            decimals=decimals,
        )

        cvar_value = cvar_expected_shortfall_historical(
            returns_array,
            alpha=alpha,
            decimals=decimals,
        )

        report[alpha] = {
            "VaR": var_value,
            "CVaR": cvar_value,
        }

    return report


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    simulated_daily_returns = rng.normal(0.0005, 0.01, size=2000)
    simulated_daily_returns_array = _as_1d_float_array(simulated_daily_returns)

    print(
        "VaR 5%:",
        value_at_risk_historical(simulated_daily_returns_array, alpha=0.05, decimals=4),
    )

    print(
        "CVaR 5%:",
        cvar_expected_shortfall_historical(simulated_daily_returns_array, alpha=0.05, decimals=4),
    )

    print(
        "Report:",
        var_cvar_report(simulated_daily_returns_array, alpha_levels=(0.01, 0.05, 0.10), decimals=4),
    )


