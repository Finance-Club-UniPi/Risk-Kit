import numpy as np
import yfinance as yf

def jensens_alpha(portofolio_returns, market_returns, risk_free_rate):
    portofolio_returns = np.array(portofolio_returns)
    market_returns = np.array(market_returns)
    Rp = np.mean(portofolio_returns)
    Rm = np.mean(market_returns)
    covariance = np.cov(portofolio_returns, market_returns)[0,1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    alpha = Rp - ( risk_free_rate + beta * (Rm - risk_free_rate))
    return alpha, beta
stocks = {
    "Nike": "NKE",
    "NVIDIA": "NVDA",
    "Plug Power": "PLUG",
    "Tesla": "TSLA"
}
market_ticker = "^GSPC"
start_date = "2019-01-01"
end_date = "2024-01-01"
annual_rf = 0.02
risk_free_rate = annual_rf / 252
market_data = yf.download(market_ticker, start=start_date, end=end_date, auto_adjust=False)
print(market_data.head())
market_returns = market_data["Adj Close"].pct_change().dropna()
print (market_returns.head())
print("\n====== JENSEN'S ALPHA: LUXURY BRANDS ======\n")
for name, ticker in stocks.items():
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    print(stock_data.head())
    stock_returns = stock_data["Adj Close"].pct_change().dropna()
    print(stock_returns.head())
    common_dates = stock_returns.index.intersection(market_returns.index)
    stock_returns = stock_returns.loc[common_dates]
    market_returns_aligned = market_returns.loc[common_dates]
    alpha, beta = jensens_alpha(stock_returns, market_returns_aligned, risk_free_rate)
    print(name)
    print(f"  Beta: {beta:.3f}")
    print(f"  Jensen's Alpha (daily): {alpha:.6f}\n")

