import yfinance as yf


def calculate_var(returns, portfolio_value=10000, confidence=0.95):
    sorted_returns = sorted(returns)
    position = min(int((1 - confidence) * len(returns)), len(returns) - 1)
    var_return = abs(sorted_returns[position])
    var_amount = var_return * portfolio_value
    var_percentage = var_return * 100
    return var_amount, var_percentage
    
if __name__ == "__main__":
    close_prices = yf.download("^SP500TR", period="max", progress=False, auto_adjust=True)["Close"]
    returns = close_prices.pct_change().dropna().values.flatten()
    portfolio = 25000

    var_95_amount, var_95_percent = calculate_var(returns, portfolio, 0.95)
    var_99_amount, var_99_percent = calculate_var(returns, portfolio, 0.99)

    print("VALUE AT RISK CALCULATION")
    print("=" * 40)
    print(f"Portfolio value: ${portfolio:,}")
    print(f"Number of data points: {len(returns)}\n")
    print(f"95% Daily VaR: ${var_95_amount:,.2f} ({var_95_percent:.2f}%)")
    print(f"99% Daily VaR: ${var_99_amount:,.2f} ({var_99_percent:.2f}%)\n")
    print(f"95% confidence: Won't lose more than ${var_95_amount:,.2f} in one day")
    print(f"99% confidence: Won't lose more than ${var_99_amount:,.2f} in one day")
   