# Markowitz Portfolio Optimization

This module implements Modern Portfolio Theory (MPT), developed by Harry Markowitz in 1952. MPT is a mathematical framework for constructing portfolios that maximize expected return for a given level of risk, or minimize risk for a given expected return.

## Understanding Portfolio Theory

### The Core Idea

When you invest in multiple assets, you don't just get the average return. You also benefit from diversification. If two assets don't move perfectly together, combining them can reduce overall risk. Markowitz showed how to mathematically find the optimal mix of assets.

### Key Concepts

**Portfolio Return**: If you invest 40% in Asset A (returning 8%) and 60% in Asset B (returning 12%), your portfolio return is:
- $0.4 \times 0.08 + 0.6 \times 0.12 = 0.104$ (or 10.4%)

**Portfolio Risk**: Risk isn't just the average of individual asset risks. If assets move together, risk compounds. If they move in opposite directions, risk cancels out. This is captured by the covariance matrix.

**Efficient Frontier**: A curve showing all portfolios that offer the best possible return for each level of risk. Any portfolio below this curve is inefficient: you could get more return for the same risk, or less risk for the same return.

## How the Functions Work

### `estimate_mean_cov(returns, annualization_factor=252)`

**What it does**: Takes historical return data and estimates two key statistics needed for portfolio optimization: expected returns and the covariance matrix.

**How it works**:

1. **Input validation**: First, it converts your input to a 2D NumPy array. If you give it a 1D array (single asset), it reshapes it to have one column. It checks that you have at least 2 time periods (you can't compute variance from a single observation).

2. **Computing expected returns**: It calculates the mean return for each asset across all time periods. This gives you the average daily return. Then it multiplies by the `annualization_factor` (default 252 for trading days) to convert to annual returns.

   For example, if Asset A has an average daily return of 0.0005 (0.05%), the annualized return is:
   - $0.0005 \times 252 = 0.126$ (12.6% per year)

3. **Computing covariance**: It uses NumPy's `np.cov()` function with `rowvar=False` (meaning each column is a variable/asset) and `ddof=1` (degrees of freedom correction for sample variance). This creates a matrix where:
   - Diagonal elements are each asset's variance
   - Off-diagonal elements show how pairs of assets move together (covariance)
   
   The covariance is also annualized by multiplying by the annualization factor.

4. **Validation**: It checks that the covariance isn't a scalar (which would happen with only one asset). The module requires at least 2 assets because portfolio optimization is about diversification.

**Why annualization matters**: Daily returns are small numbers (like 0.0005). Annualizing makes them more interpretable (12.6% per year) and ensures all your statistics are on the same time scale.

**Example**:
```python
import numpy as np
from markowitz.Markowitz import estimate_mean_cov

# Simulate 1000 days of returns for 3 assets
rng = np.random.default_rng(42)
returns = rng.normal(0.0005, 0.01, size=(1000, 3))

# Estimate statistics
expected_returns, cov = estimate_mean_cov(returns, annualization_factor=252)
```

---

### `portfolio_return(weights, expected_returns)`

**What it does**: Calculates the expected return of a portfolio given the weights (how much you invest in each asset) and the expected returns of individual assets.

**How it works**:

1. **Input preparation**: Converts both inputs to 1D NumPy arrays and ensures they're the same length.

2. **The calculation**: It performs a dot product (matrix multiplication) between weights and expected returns:
   $$
   \mu_p = \mathbf{w}^T \boldsymbol{\mu} = w_1 \mu_1 + w_2 \mu_2 + \ldots + w_n \mu_n
   $$
   
   This is just a weighted average: multiply each asset's return by its weight, then sum.

3. **Why it's simple**: Portfolio return is linear. It's just the weighted average of individual returns. Risk is more complex because of correlations.

**Example**:
```python
from markowitz.Markowitz import portfolio_return

weights = np.array([0.4, 0.3, 0.3])  # 40%, 30%, 30%
expected_returns = np.array([0.08, 0.12, 0.10])  # 8%, 12%, 10%

port_return = portfolio_return(weights, expected_returns)
# Result: 0.4*0.08 + 0.3*0.12 + 0.3*0.10 = 0.098 (9.8%)
```

---

### `portfolio_volatility(weights, cov)`

**What it does**: Calculates the volatility (standard deviation) of portfolio returns, accounting for how assets move together.

**How it works**:

1. **Input validation**: Checks that the covariance matrix is 2D (not a scalar) and square, and that dimensions match the weights.

2. **The calculation**: Portfolio variance is computed as:
   $$
   \sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}
   $$
   
   This matrix multiplication does three things:
   - Multiplies weights by the covariance matrix: $\boldsymbol{\Sigma} \mathbf{w}$
   - Then multiplies the result by weights again: $\mathbf{w}^T (\boldsymbol{\Sigma} \mathbf{w})$
   - This captures both individual asset variances (diagonal of $\boldsymbol{\Sigma}$) and correlations (off-diagonal)

3. **Why it's more complex**: Unlike return, volatility isn't just a weighted average. If two assets are perfectly correlated, portfolio risk is the weighted average of individual risks. If they're negatively correlated, risk is reduced. The covariance matrix captures all these relationships.

4. **Final step**: Takes the square root to convert variance to volatility (standard deviation), and ensures the result is non-negative (handles numerical errors).

**Example**:
```python
from markowitz.Markowitz import portfolio_volatility

weights = np.array([0.4, 0.3, 0.3])
cov = np.array([[0.04, 0.01, 0.02],    # Asset 1 variance = 0.04
                [0.01, 0.09, 0.03],    # Asset 2 variance = 0.09
                [0.02, 0.03, 0.06]])   # Asset 3 variance = 0.06

port_vol = portfolio_volatility(weights, cov)
```

---

### `global_minimum_variance_portfolio(cov)`

**What it does**: Finds the portfolio with the absolute minimum possible variance (risk), regardless of return. This is the leftmost point on the efficient frontier.

**How it works**:

1. **The optimization problem**: We want to minimize portfolio variance $\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$ subject to the constraint that weights sum to 1 ($\mathbf{1}^T \mathbf{w} = 1$).

2. **The solution**: Using calculus (Lagrange multipliers), the solution is:
   $$
   \mathbf{w}_{GMVP} = \frac{\boldsymbol{\Sigma}^{-1} \mathbf{1}}{\mathbf{1}^T \boldsymbol{\Sigma}^{-1} \mathbf{1}}
   $$
   
   Let's break this down:
   - $\boldsymbol{\Sigma}^{-1}$ is the inverse of the covariance matrix
   - $\mathbf{1}$ is a vector of ones (same length as number of assets)
   - The numerator $\boldsymbol{\Sigma}^{-1} \mathbf{1}$ gives "raw" weights
   - The denominator $\mathbf{1}^T \boldsymbol{\Sigma}^{-1} \mathbf{1}$ is a scalar that normalizes the weights to sum to 1

3. **Why it works**: The inverse covariance matrix "undoes" the correlations. Assets with high variance get lower weights, and the normalization ensures weights sum to 1.

4. **Implementation**: The function uses `np.linalg.pinv()` (pseudo-inverse) instead of regular inverse. This handles cases where the covariance matrix might be singular (not invertible) due to numerical issues or perfectly correlated assets.

**Example**:
```python
from markowitz.Markowitz import global_minimum_variance_portfolio

cov = np.array([[0.04, 0.01, 0.02],
                [0.01, 0.09, 0.03],
                [0.02, 0.03, 0.06]])

gmvp_weights = global_minimum_variance_portfolio(cov)
# Result: weights that minimize variance, summing to 1.0
```

---

### `efficient_frontier_weights(expected_returns, cov, target_return)`

**What it does**: Finds the portfolio on the efficient frontier that achieves a specific target return while minimizing risk.

**How it works**:

1. **The optimization problem**: Minimize portfolio variance $\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$ subject to:
   - Portfolio return equals target: $\mathbf{w}^T \boldsymbol{\mu} = \mu_{target}$
   - Weights sum to 1: $\mathbf{1}^T \mathbf{w} = 1$

2. **Using Lagrange multipliers**: This is a constrained optimization problem solved with Lagrange multipliers $\lambda$ and $\gamma$:
   - $\lambda$ is associated with the return constraint
   - $\gamma$ is associated with the weights-sum-to-one constraint

3. **The solution**: The optimal weights are:
   $$
   \mathbf{w} = \boldsymbol{\Sigma}^{-1} (\lambda \mathbf{1} + \gamma \boldsymbol{\mu})
   $$
   
   To find $\lambda$ and $\gamma$, we solve a system of equations:
   - $a = \mathbf{1}^T \boldsymbol{\Sigma}^{-1} \mathbf{1}$ (scalar)
   - $b = \mathbf{1}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}$ (scalar)
   - $c = \boldsymbol{\mu}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}$ (scalar)
   - Determinant: $D = ac - b^2$
   - $\lambda = \frac{c - b \mu_{target}}{D}$
   - $\gamma = \frac{a \mu_{target} - b}{D}$

4. **Why this works**: The formula balances two forces:
   - The $\lambda \mathbf{1}$ term ensures weights sum to 1
   - The $\gamma \boldsymbol{\mu}$ term ensures the target return is achieved
   - The inverse covariance $\boldsymbol{\Sigma}^{-1}$ minimizes risk

5. **Validation**: The function checks that the determinant $D$ isn't too close to zero (which would indicate a degenerate system, perhaps because all assets have the same return, making the problem unsolvable).

**Example**:
```python
from markowitz.Markowitz import efficient_frontier_weights

expected_returns = np.array([0.08, 0.12, 0.10])
cov = np.array([[0.04, 0.01, 0.02],
                [0.01, 0.09, 0.03],
                [0.02, 0.03, 0.06]])

target = 0.10  # Target 10% return
weights = efficient_frontier_weights(expected_returns, cov, target)
# Result: weights that achieve 10% return with minimum risk
```

---

### `tangency_portfolio(expected_returns, cov, risk_free_rate=0.0)`

**What it does**: Finds the portfolio that maximizes the Sharpe ratio, which is the excess return (above risk-free rate) per unit of risk. This is the portfolio where a line from the risk-free rate is tangent to the efficient frontier.

**How it works**:

1. **The Sharpe ratio**: Defined as:
   $$
   S = \frac{\mu_p - r_f}{\sigma_p}
   $$
   where $r_f$ is the risk-free rate. Higher Sharpe ratio = better risk-adjusted return.

2. **The optimization**: To maximize Sharpe ratio, we maximize $\frac{\mathbf{w}^T (\boldsymbol{\mu} - r_f)}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}}$.

3. **The solution**: The optimal weights (before normalization) are:
   $$
   \mathbf{w}_{raw} = \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu} - r_f)
   $$
   
   This uses excess returns ($\boldsymbol{\mu} - r_f$) instead of raw returns. Assets with higher excess returns relative to their risk get more weight.

4. **Normalization**: The raw weights are normalized so they sum to 1:
   $$
   \mathbf{w} = \frac{\mathbf{w}_{raw}}{\mathbf{1}^T \mathbf{w}_{raw}}
   $$
   
   This ensures it's a fully invested portfolio.

5. **Why it's special**: The tangency portfolio is the optimal risky portfolio when you can also invest in a risk-free asset. It's the single best portfolio for combining with cash/bonds.

**Example**:
```python
from markowitz.Markowitz import tangency_portfolio

expected_returns = np.array([0.08, 0.12, 0.10])
cov = np.array([[0.04, 0.01, 0.02],
                [0.01, 0.09, 0.03],
                [0.02, 0.03, 0.06]])

risk_free_rate = 0.02  # 2% risk-free rate
tangency_weights = tangency_portfolio(expected_returns, cov, risk_free_rate)
# Result: weights that maximize Sharpe ratio
```

---

### `efficient_frontier(expected_returns, cov, n_points=25)`

**What it does**: Generates multiple portfolios along the efficient frontier by computing optimal weights for a range of target returns.

**How it works**:

1. **Creating target returns**: It creates a grid of target returns from the minimum expected return (most conservative) to the maximum expected return (most aggressive) among the assets.

2. **Looping through targets**: For each target return, it:
   - Calls `efficient_frontier_weights()` to get optimal weights
   - Computes the actual portfolio return using `portfolio_return()`
   - Computes the portfolio volatility using `portfolio_volatility()`
   - Stores all three (return, volatility, weights)

3. **Why generate multiple points**: The efficient frontier is a curve, not a single point. By computing many points, you can visualize the trade-off between risk and return.

4. **Output**: Returns three arrays/lists:
   - Returns: portfolio returns for each point
   - Volatilities: portfolio volatilities for each point
   - Weights: the weight vectors for each point

**Example**:
```python
from markowitz.Markowitz import efficient_frontier, estimate_mean_cov
import matplotlib.pyplot as plt

# Generate sample data
rng = np.random.default_rng(42)
returns = rng.normal(0.0005, 0.01, size=(1000, 3))
expected_returns, cov = estimate_mean_cov(returns)

# Generate efficient frontier
port_returns, port_vols, weights_list = efficient_frontier(
    expected_returns, cov, n_points=50
)

# Plot it
plt.figure(figsize=(10, 6))
plt.plot(port_vols, port_returns, 'b-', linewidth=2, label='Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend()
plt.grid(True)
plt.show()
```

## Common Workflows

### Basic Portfolio Analysis

```python
import numpy as np
from markowitz.Markowitz import (
    estimate_mean_cov,
    portfolio_return,
    portfolio_volatility,
    global_minimum_variance_portfolio,
    tangency_portfolio
)

# Step 1: Load your returns data (time periods × assets)
returns = np.array([...])  # Your data here

# Step 2: Estimate statistics
expected_returns, cov = estimate_mean_cov(returns)

# Step 3: Find minimum variance portfolio
gmvp_weights = global_minimum_variance_portfolio(cov)
gmvp_return = portfolio_return(gmvp_weights, expected_returns)
gmvp_vol = portfolio_volatility(gmvp_weights, cov)

# Step 4: Find tangency portfolio (max Sharpe)
risk_free_rate = 0.02
tangency_weights = tangency_portfolio(expected_returns, cov, risk_free_rate)
tangency_return = portfolio_return(tangency_weights, expected_returns)
tangency_vol = portfolio_volatility(tangency_weights, cov)
sharpe_ratio = (tangency_return - risk_free_rate) / tangency_vol
```

## Important Notes

### Why We Reject Single-Asset Portfolios

The module requires at least 2 assets because:
- Portfolio optimization is about diversification
- With one asset, there's no portfolio to optimize. You just hold that asset.
- The covariance matrix becomes a scalar (single number), which breaks matrix operations

### Annualization

The `annualization_factor` converts daily/weekly/monthly statistics to annual:
- Daily data: use 252 (trading days per year)
- Weekly data: use 52
- Monthly data: use 12

This ensures all statistics are on the same time scale for comparison.

### Weight Constraints

- `global_minimum_variance_portfolio` and `tangency_portfolio`: Weights always sum to 1 (fully invested)
- `efficient_frontier_weights`: Weights may not sum to 1 if short selling is allowed (negative weights)

### Error Messages

The module provides clear error messages:
- **"Single-asset portfolios are not supported"**: You need at least 2 assets
- **"Dimension mismatch"**: Arrays have incompatible sizes
- **"Covariance must be square"**: Covariance matrix must be N×N for N assets
- **"Degenerate system"**: The optimization problem can't be solved (e.g., all assets have identical returns)

## References

- Markowitz, H. (1952). "Portfolio Selection". *The Journal of Finance*, 7(1), 77-91.
- Markowitz, H. (1959). *Portfolio Selection: Efficient Diversification of Investments*. John Wiley & Sons.
