import numpy as np
import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt

close_prices = yf.download("^SP500TR", start="1985-01-01", end="2025-01-01", progress=False, auto_adjust=True)['Close']
print(close_prices.head(2))


returns = close_prices.pct_change().dropna().values.flatten()
nobs = len(returns)
nboot = 55000  # number of bootstrap samples

INITIAL_WEALTH = 1000  # start with $1000

boot_returns = np.random.choice(
    returns,
    size=(nboot, nobs),
    replace = True)

terminal_wealth = INITIAL_WEALTH * np.prod(1 + boot_returns, axis=1)

mean_terminal_wealth = terminal_wealth.mean()
q05_terminal_wealth = np.percentile(terminal_wealth, 5)

plt.hist(terminal_wealth, bins=3000, density=True)
plt.axvline(INITIAL_WEALTH, color='red', linestyle='--', label=f'Initial Wealth (${INITIAL_WEALTH})')
plt.axvline(mean_terminal_wealth, color='green', linestyle='-', label=f'Mean Terminal Wealth (${mean_terminal_wealth:.0f})')
plt.axvline(q05_terminal_wealth, color='orange', linestyle=':', label=f'5% Percentile (${q05_terminal_wealth:.0f})')
plt.xlabel('Terminal Wealth ($)')
plt.ylabel("Density")
plt.legend()
plt.show()
print(f'Initial wealth: ${INITIAL_WEALTH}')
print(f'5% percentile of terminal wealth: ${q05_terminal_wealth:.2f}')
print(f'Mean terminal wealth: ${mean_terminal_wealth:.2f}')
