def calculate_var(returns, portfolio_value=10000, confidence=0.95):
    sorted_returns = sorted(returns)
    position = int((1 - confidence) * len(returns))
    var_return = abs(sorted_returns[position])
    var_amount = var_return * portfolio_value
    var_percentage = var_return * 100
    return var_amount, var_percentage
if __name__ == "__main__":
    example_returns = [0.012, -0.023, 0.015, -0.018, 0.007,-0.025, 0.009, -0.021, 0.011, -0.016,0.013, -0.019, 0.008, -0.022, 0.006,-0.017, 0.014, -0.024, 0.010, -0.020]
    portfolio = 25000

    var_95_amount, var_95_percent = calculate_var(example_returns, portfolio, 0.95)
    var_99_amount, var_99_percent = calculate_var(example_returns, portfolio, 0.99)

    print("VALUE AT RISK CALCULATION")
    print("=" * 40)
    print("Portfolio value: ${portfolio:,}")
    print("Number of data points: ${len(example_returns)}\n")
    print("95% Daily VaR: ${var_95_amount:,.2f} ({var_95_percent:.2f}%)")
    print("99% Daily VaR: ${var_99_amount:,.2f} ({var_99_percent:.2f}%)\n")
    print("95% confidence: Won't lose more than ${var_95_amount:,.2f} in one day")
    print("99% confidence: Won't lose more than ${var_99_amount:,.2f} in one day")
   