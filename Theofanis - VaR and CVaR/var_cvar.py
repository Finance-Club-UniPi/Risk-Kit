import numpy as np
import yfinance as yf


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


def fetch_monthly_returns(symbol, period="20y"):
    price_data = yf.download(
        symbol,
        period=period,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )

    if price_data.empty:
        raise ValueError(f"No data downloaded for symbol '{symbol}'.")

    close_prices = price_data["Close"].dropna()

    if close_prices.empty:
        raise ValueError(f"No valid closing prices for symbol '{symbol}'.")

    monthly_returns = close_prices.pct_change().dropna().to_numpy()
    return _as_1d_float_array(monthly_returns)


if __name__ == "__main__":
    symbol = "SPY"
    monthly_returns = fetch_monthly_returns(symbol=symbol, period="20y")

    alpha_levels = (0.01, 0.05, 0.10)
    report = var_cvar_report(monthly_returns, alpha_levels=alpha_levels)

    print(f"\nHistorical Risk Report (Monthly Returns from {symbol})")
    print("-" * 56)
    print(f"Sample size: {monthly_returns.size} observations")
    print(f"Mean monthly return: {np.mean(monthly_returns) * 100:.3f}%")
    print(f"Monthly volatility (std dev): {np.std(monthly_returns) * 100:.3f}%")
    print()
    print("Interpretation:")
    print("- VaR: minimum loss you expect to exceed only alpha fraction of months.")
    print("- CVaR: average loss on those worst-case tail months.")
    print()
    print("Alpha   Confidence   VaR (loss)   CVaR (loss)")
    print("-" * 56)

    for alpha in alpha_levels:
        confidence = 1.0 - alpha
        var_loss = report[float(alpha)]["VaR"]
        cvar_loss = report[float(alpha)]["CVaR"]
        print(
            f"{alpha:>5.0%}   {confidence:>10.0%}   "
            f"{var_loss * 100:>9.2f}%   {cvar_loss * 100:>10.2f}%"
        )
    print()
