import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

def calculate_cvar(weights, returns, alpha=0.05):
    portfolio_returns = returns.dot(weights)
    var_cutoff = np.percentile(portfolio_returns, alpha * 100)
    tail_losses = portfolio_returns[portfolio_returns <= var_cutoff]
    
    if len(tail_losses) == 0:
        return 0.0
    
    cvar = tail_losses.mean()
    return -cvar

def rockafellar_objective(params, returns, alpha=0.05):
    zeta = params[0]
    weights = params[1:]
    
    portfolio_return = returns.dot(weights)
    loss = -portfolio_return
    
    hinge_loss = np.maximum(0, loss - zeta)
    cvar = zeta + (1 / (alpha * len(returns))) * np.sum(hinge_loss)
    
    return cvar

def minimize_cvar_ru(returns, alpha=0.05):
    num_assets = returns.shape[1]
    
    init_guess = np.concatenate(([0.0], np.repeat(1/num_assets, num_assets)))
    
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda x: np.sum(x[1:]) - 1}
    bounds = [(None, None)] + [(0, 1) for _ in range(num_assets)]
    
    result = minimize(rockafellar_objective, 
                      init_guess, 
                      args=(returns, alpha), 
                      method='SLSQP', 
                      bounds=bounds, 
                      constraints=(weights_sum_to_1,))
    
    if result.success:
        optimal_weights = result.x[1:]
        return optimal_weights
    else:
        raise ValueError("Optimization failed: " + result.message)

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    
    raw_data = yf.download(tickers, start="2022-01-01", end="2024-01-01")
    
    if raw_data.empty: 
        print("Download failed.")
        exit()
        
    if 'Adj Close' in raw_data.columns:
        data = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        data = raw_data['Close']
    else:
        print("Column not found.")
        exit()
        
    real_returns = data.pct_change().dropna()

    optimal_weights = minimize_cvar_ru(real_returns)

    print("Optimal Weights:")
    for stock, weight in zip(tickers, optimal_weights):
        print(f"{stock}: {weight*100:.2f}%")
        
    cvar_result = calculate_cvar(optimal_weights, real_returns)
    print(f"\nMinimum CVaR : {cvar_result:.4f}")