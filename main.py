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
import scypi.optimize as opt

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

    print(f"The returns are {returns:.4f}")
    print(f"The standard deviation is {std_dev:.4f} ")
    print(f"The sharpe ratio is {sharpe:.4f}")

# TEST ABOVE FUNC WITH DUMMY DATA 
test_weights = np.array([0.5, 0.5])
test_means = np.array([0.10, 0.20]) 
rf_rate = 0.0
test_cov = np.array([
    [0.04, 0.00], 
    [0.00, 0.04]])
portfolio_returns(test_weights, test_means, test_cov, rf_rate)
