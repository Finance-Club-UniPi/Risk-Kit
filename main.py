print("hello world!")

x = 10
if x > 9:
    print("Yes!")
elif x < 0:
    print("No!")

print("hi!")

print("hi this is my branch")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

tickers = ['^GSPC','NVDA']

data = yf.download(tickers, period='25y', interval='1mo',auto_adjust=True)
prices = data['Close']
returns = prices.pct_change().dropna()
# TEST CELL
print(returns.head())
min_weight = 0.2
rf_rate = 0.04
# MEAN MONTHLY RETURNS AND COVARIANCE ANNUALIZED
mean_returns = returns.mean() * 12
cov_matrix = returns.cov() * 12
# FUNCTION THAT CALCULATES PORTFOLIO RETURN, VOLATILITY & SHARPE RATIO
def portfolio_returns(weights,mean_returns,cov_matrix,rf_rate):
    
    weights = np.array(weights)
    returns = np.sum(mean_returns*weights)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (returns - rf_rate) / std_dev

    return returns, std_dev, sharpe

# TEST ABOVE FUNC WITH DUMMY DATA 
test_weights = np.array([0.5, 0.5])
test_means = np.array([0.10, 0.20]) 
rf_rate = 0.0
test_cov = np.array([
    [0.04, 0.00], 
    [0.00, 0.04]])
portfolio_returns(test_weights, test_means, test_cov, rf_rate)

# FINDS THE HIGHEST SHARPE RATIO NUMBER (SCYPI CAN ONLY FIND THE SMALLEST SO THATS WHY IT IS LIKE THAT)
def neg_sharpe(weights, mean_returns, cov_matrix, rf_rate):
    return -portfolio_returns(weights, mean_returns, cov_matrix, rf_rate)[2]

# MINIZE VOLATILITY AND FIND THE SAFEST ONE
def minimize_vol(weights, mean_returns, cov_matrix, rf_rate):
    return portfolio_returns(weights, mean_returns, cov_matrix, rf_rate)[1]

# MONTE CARLO SIMULATION 
num_portfolios = 10000
all_weights = np.zeros((num_portfolios, len(tickers)))   # CREATE SPACE FOR THE DATA WHICH WILL BE CALCULATED
ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)

print(f"Simulating {num_portfolios} portfolios with MIN WEIGHT {min_weight}")

# SIMULATION
for i in range(num_portfolios):
    weights = np.random.random(len(tickers))

    # Normalize the random numbers so they sum to the "remaining space" (0.6)
    remaining_space = 1.0 - (min_weight * len(tickers))
    weights = weights / np.sum(weights) * remaining_space

    # Add the base minimum weight (0.2) to everyone
    weights = weights + min_weight

    all_weights[i,:] = weights

    # CALCULATE PERFORMANCE
    p_ret, p_vol, p_sharpe = portfolio_returns(weights, mean_returns, cov_matrix, rf_rate)

    # STORE RESULTS IN THE ARRAYS
    ret_arr[i] = p_ret
    vol_arr[i] = p_vol
    sharpe_arr[i] = p_sharpe

print("Simulation Complete!")
print(f"Generated {num_portfolios} portfolios. All respect the {min_weight*100}% min weight constraint.")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Markowitz Efficient Frontier (Monte Carlo)')
plt.show()

# Find the index of the best portfolio
max_idx = sharpe_arr.argmax() # SCANS THE POSITION OF THE HIGHEST NUMBER

print("TOP PORTFOLIO RESULTS:")
print(f"Return: {ret_arr[max_idx]:.4f}")
print(f"Volatility: {vol_arr[max_idx]:.4f}")
print(f"Sharpe Ratio: {sharpe_arr[max_idx]:.4f}")
print("\nWEIGHTS:")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {all_weights[max_idx, i]*100:.2f}%")

# OBJERVATIONS 
print("How is it a straight line??")
print("\nWell it's not a complete straight line because it has a very small curve. So why is that?")
print("\nThis happens because first of all we have 2 assets whose corelation is extremely close to 1 but not exactly1.")
print("\nAlso the fact that we have placed a constraint of a minimum weight = 20% on the simulation makes the curve even smaller")
print("\nOn the one hand NVIDIA has a higher volatility (60%) and astronomical returns (4000%) compared to the simpler SP500 (vol:15%, ret:10%)")
print("\nThis is very common on portfolios with a broad index and a high volatility/high return stock over a long period of time")
print("\nThe slight curve happens at the Minimum Variance Portfolio (safest one) where NVIDIA takes the min wright (20%)")
