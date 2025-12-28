import numpy as np


def _as_1d_float_array(returns):
    returns_array = np.asarray(returns, dtype=float).reshape(-1)

    if returns_array.size == 0:
        raise ValueError("Input returns are empty.")

    return returns_array


def wealth_index(portfolio_returns, start_value=1.0):
    portfolio_returns = _as_1d_float_array(portfolio_returns)

    if start_value <= 0:
        raise ValueError("start_value must be > 0.")

    portfolio_values = start_value * np.cumprod(1.0 + portfolio_returns)

    return portfolio_values


def drawdown_series(portfolio_returns, start_value=1.0):
    portfolio_values = wealth_index(portfolio_returns, start_value=start_value)

    running_peaks = np.maximum.accumulate(portfolio_values)

    drawdowns = (running_peaks - portfolio_values) / running_peaks

    return drawdowns


def max_drawdown(portfolio_returns, start_value=1.0):
    drawdowns = drawdown_series(portfolio_returns, start_value=start_value)

    maximum_drawdown = np.max(drawdowns)

    return float(maximum_drawdown)


def max_drawdown_duration(portfolio_returns, start_value=1.0):
    portfolio_values = wealth_index(portfolio_returns, start_value=start_value)

    peak_value = portfolio_values[0]

    current_drawdown_duration = 0
    longest_drawdown_duration = 0

    for value in portfolio_values:
        if value >= peak_value:
            peak_value = value
            current_drawdown_duration = 0
        else:
            current_drawdown_duration += 1
            longest_drawdown_duration = max(longest_drawdown_duration, current_drawdown_duration)

    return int(longest_drawdown_duration)


def drawdown_diagnostics(portfolio_returns, start_value=1.0):
    portfolio_returns = _as_1d_float_array(portfolio_returns)

    portfolio_values = wealth_index(portfolio_returns, start_value=start_value)

    running_peaks = np.maximum.accumulate(portfolio_values)

    drawdowns = (running_peaks - portfolio_values) / running_peaks

    trough_index = int(np.argmax(drawdowns))

    maximum_drawdown = float(drawdowns[trough_index])

    peak_index_before_trough = int(np.argmax(portfolio_values[: trough_index + 1]))

    diagnostics = {
        "max_drawdown": maximum_drawdown,
        "peak_index": peak_index_before_trough,
        "trough_index": trough_index,
        "max_drawdown_duration": max_drawdown_duration(portfolio_returns, start_value=start_value),
    }

    return diagnostics


if __name__ == "__main__":
    example_returns = [0.01, -0.02, 0.03, -0.10, 0.05, 0.02, -0.01, 0.04]

    print(max_drawdown(example_returns))
    print(max_drawdown_duration(example_returns))
    print(drawdown_diagnostics(example_returns))
