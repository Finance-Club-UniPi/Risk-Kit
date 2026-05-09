import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib.pyplot as plt

#download data
def data_download(assets, prd):
    prices = yf.download(assets, period = prd, auto_adjust = True)['Close']
    returns = prices.pct_change().dropna()
    return returns

#portfolio statistics calculation
def historical_statistics(returns):
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    return mean_returns, cov_matrix

#monte carlo simulation
def monte_carlo_sim(mean_returns, cov_matrix, sims, degrees_of_freedom):
    n = len(mean_returns)

    Z = np.random.multivariate_normal(np.zeros(n), cov_matrix, sims) 

    chi2_samples = chi2.rvs(degrees_of_freedom, sims)
    scaling = np.sqrt(degrees_of_freedom/chi2_samples).reshape(-1,1)

    sim_returns = mean_returns + Z * scaling
    return sim_returns

#portfolio returns
def portfolio_returns(sim_returns, weights):
    return sim_returns @ weights

#calculate portfolio volatility
def volatility(cov_matrix, weights):
    variance = weights.T @ cov_matrix @ weights #portfolio variance
    std_dev = np.sqrt(variance) #portfolio volatility / standard deviation
    return std_dev

def var_cvar(portfolio_returns, alpha): #alpha = confidence level
    losses = -portfolio_returns #convert returns to losses to calculate the VaR
    losses_sorted = np.sort(losses)
    n = len(losses_sorted)
    k = int(np.ceil(alpha * n)) - 1 #VaR index
    
    #Calculate the VaR
    VaR = losses_sorted[k]

    #calculate the CVaR
    coeff = 1 / (n- k) #CVaR coefficient
    CVaR = coeff * losses_sorted[k:].sum()

    return VaR, CVaR

def equal_weights(assets):
    n = len(assets)
    weights = np.ones(n)/n
    return weights

def optimal_weights(mean_returns, cov_matrix, risk_free):
    n = len(mean_returns)

    def negative_sharpe(weights):
        ret = portfolio_returns(mean_returns, weights)
        vol = volatility(cov_matrix, weights)
        return -(ret-risk_free)/vol
    
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w)-1})

    bounds = tuple((0,1) for _ in range(n)) #weights must be between 0 and 1

    result = minimize(negative_sharpe, x0 = np.ones(n)/n, bounds = bounds, constraints = constraints, method = "SLSQP")

    return result.x

#plotting

def plot_var_cvar(portfolio_returns, VaR, CVaR, title):
    plt.figure(figsize=(10,6))
    plt.hist(portfolio_returns, bins=200, density=True)
    
    plt.axvline(-VaR, linestyle='--', linewidth=2, label=f"VaR")
    plt.axvline(-CVaR, linestyle='-', linewidth=2, label=f"CVaR")
    
    plt.xlabel("Portfolio Return")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_return_comparison(eq_portfolio, M_portfolio):
    plt.figure(figsize=(10,6))
    plt.hist(eq_portfolio, bins=200, density=True, alpha=0.5, label="Equal Weight")
    plt.hist(M_portfolio, bins=200, density=True, alpha=0.5, label="Markowitz")
    plt.xlabel("Portfolio Return")
    plt.ylabel("Density")
    plt.title("Simulated Return Distribution Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cdf(eq_portfolio, M_portfolio):
    def cdf(data):
        sorted_data = np.sort(data)
        return sorted_data, np.arange(1, len(sorted_data)+1)/len(sorted_data)

    eq_x, eq_y = cdf(eq_portfolio)
    M_x, M_y = cdf(M_portfolio)

    plt.figure(figsize=(10,6))
    plt.plot(eq_x, eq_y, label="Equal Weight")
    plt.plot(M_x, M_y, label="Markowitz")
    plt.xlabel("Portfolio Return")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Simulated Portfolio Returns")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_worst_tail(eq_portfolio, M_portfolio, tail=0.01):
    eq_tail = np.sort(eq_portfolio)[:int(tail * len(eq_portfolio))]
    M_tail = np.sort(M_portfolio)[:int(tail * len(M_portfolio))]

    plt.figure(figsize=(10,6))
    plt.hist(eq_tail, bins=100, density=True, alpha=0.5, label="Equal Weight")
    plt.hist(M_tail, bins=100, density=True, alpha=0.5, label="Markowitz")
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.title(f"Worst {int(tail*100)}% Simulated Portfolio Returns")
    plt.legend()
    plt.grid(True)
    plt.show()

#example

if __name__ == "__main__":
    #assumptions
    assets = ["AAPL", "TSLA"]
    sims = 1000000
    alpha = 0.99
    risk_free = 0
    dof = 5 #degrees of freedom

    #functions called
    returns = data_download(assets, '3y') #data download for AAPL, TSLA for 3 years

    returns_mean , returns_covar = historical_statistics(returns) #mean and covariance matrix of the data

    sim_returns = monte_carlo_sim(returns_mean, returns_covar, sims, degrees_of_freedom=5) #monte carlo simulation

    eq_weights = equal_weights(assets) #example weights

    eq_portfolio = portfolio_returns(sim_returns, eq_weights) #example portfolio

    M_weights = optimal_weights(returns_mean, returns_covar, risk_free) #Markowitz portfolio weights

    M_portfolio = portfolio_returns(sim_returns, M_weights) #Markowitz portfolio returns

    eq_VaR, eq_CVaR = var_cvar(eq_portfolio, alpha) #VaR, CVaR calculation of example portfolio
    eq_vol = volatility(returns_covar, eq_weights) #example portfolio volatility calculation

    M_VaR, M_CVaR = var_cvar(M_portfolio, alpha) #VaR, CVaR calculation
    M_vol = volatility(returns_covar, M_weights) #markowitz portfolio volatility calculation

    #output
    print(f"Equal Weight Portfolio Value at Risk (VaR) ({alpha:.0%}): {eq_VaR:.4%}        Markowitz Portfolio Value at Risk (VaR) ({alpha:.0%}): {M_VaR:.4%}")
    print(f"Equal Weight Portfolio Conditional Value at Risk (CVaR) ({alpha:.0%}): {eq_CVaR:.4%}     Markowitz Portfolio Conditional Value at Risk (CVaR) ({alpha:.0%}): {M_CVaR:.4%}")
    print(f"Equal Weight Portfolio Volatility: {eq_vol:.4%}     Markowitz Portfolio Volatility: {M_vol:.4%}")

    plot_var_cvar(eq_portfolio, eq_VaR, eq_CVaR, "Equal Weight Portfolio – Simulated Returns with VaR & CVaR")

    plot_var_cvar(M_portfolio, M_VaR, M_CVaR, "Markowitz Portfolio – Simulated Returns with VaR & CVaR")

    plot_return_comparison(eq_portfolio, M_portfolio)

    plot_cdf(eq_portfolio, M_portfolio)

    plot_worst_tail(eq_portfolio, M_portfolio)
