import numpy as np
import yfinance as yf


def jensens_alpha(portfolio_returns, market_returns, risk_free_rate):
    portfolio_returns = np.array(portfolio_returns)
    market_returns = np.array(market_returns)

    Rp = np.mean(portfolio_returns)
    Rm = np.mean(market_returns)
    covariance = np.cov(portfolio_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)

    beta = covariance / market_variance
    alpha = Rp - (risk_free_rate + beta * (Rm - risk_free_rate))

    return alpha, beta


def download_returns(ticker, start_date, end_date):
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        interval="1mo",
    )
    return data["Adj Close"].squeeze().pct_change().dropna()



stocks = {
    "Nike": "NKE",
    "NVIDIA": "NVDA",
    "Plug Power": "PLUG",
    "Tesla": "TSLA",
}

market_ticker = "^SP500TR"
start_date = "2019-01-01"
end_date = "2026-01-01"
annual_rf = 0.03
risk_free_rate = annual_rf / 252

market_returns = download_returns(market_ticker, start_date, end_date)

print("\n====== JENSEN'S ALPHA: LUXURY BRANDS ======\n")

for name, ticker in stocks.items():
    stock_returns = download_returns(ticker, start_date, end_date)
    common_dates = stock_returns.index.intersection(market_returns.index)

    stock_returns = stock_returns.loc[common_dates]
    market_returns_aligned = market_returns.loc[common_dates]

    alpha, beta = jensens_alpha(stock_returns, market_returns_aligned, risk_free_rate)

    print(name)
    print(f"  Beta: {beta:.3f}")
    print(f"  Jensen's Alpha (monthly): {alpha:.6f}\n")

