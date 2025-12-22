import numpy as np


def _as_1d_float_array(returns):
    returns_array = np.asarray(returns, dtype=float).reshape(-1)

    if returns_array.size == 0:
        raise ValueError("Input returns are empty.")

    return returns_array


def value_at_risk_historical(portfolio_returns, alpha=0.05):
    portfolio_returns = _as_1d_float_array(portfolio_returns)

    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")

    var_threshold_return = np.quantile(portfolio_returns, alpha)

    var_loss = -var_threshold_return
    return float(var_loss)


def cvar_expected_shortfall_historical(portfolio_returns, alpha=0.05):
    portfolio_returns = _as_1d_float_array(portfolio_returns)

    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")

    var_threshold_return = np.quantile(portfolio_returns, alpha)

    tail_returns = portfolio_returns[portfolio_returns <= var_threshold_return]

    if tail_returns.size == 0:
        return float("nan")

    expected_shortfall_loss = np.mean(-tail_returns)
    return float(expected_shortfall_loss)


def var_cvar_report(portfolio_returns, alpha_levels=(0.01, 0.05)):
    portfolio_returns = _as_1d_float_array(portfolio_returns)

    report = {}

    for alpha in alpha_levels:
        alpha = float(alpha)

        report[alpha] = {
            "VaR": value_at_risk_historical(portfolio_returns, alpha=alpha),
            "CVaR": cvar_expected_shortfall_historical(portfolio_returns, alpha=alpha),
        }

    return report


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    simulated_daily_returns = rng.normal(0.0005, 0.01, size=2000)

    print(value_at_risk_historical(simulated_daily_returns, alpha=0.05))
    print(cvar_expected_shortfall_historical(simulated_daily_returns, alpha=0.05))
    print(var_cvar_report(simulated_daily_returns, alpha_levels=(0.01, 0.05, 0.10)))
