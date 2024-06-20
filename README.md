# Pairs Trading Algorithm

## Introduction

This user guide will walk you through the steps to use the pairs trading algorithm implemented in this repository. The guide includes instructions on how to set up the environment, run the algorithm, interpret trading signals, and adjust parameters for optimal results.

## Prerequisites

Ensure you have the following installed on your system:
- Python 3.x
- Required libraries: `yfinance`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`

You can install the required libraries using:
```sh
pip install yfinance pandas numpy matplotlib seaborn statsmodels
```

## Setting Up

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/your-repo/pairs-trading.git
   cd pairs-trading
   ```

2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Running the Algorithm

### Step 1: Data Collection

The script downloads historical price data for a set of financial instruments (stocks in this case) using the `yfinance` library.

```python
import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "NVDA", "BABA", "INTC", "CSCO", "ORCL", "IBM", "ADBE"]
data = yf.download(tickers, start="2010-01-01", end="2023-01-01")['Adj Close']
data.fillna(method='ffill', inplace=True)
```

### Step 2: Cointegration Analysis

Identify pairs of assets with a high degree of cointegration using the Engle-Granger two-step method.

```python
from statsmodels.tsa.stattools import coint
import numpy as np

def find_cointegrated_pairs(data, significance_level=0.10):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            S1 = data.iloc[:, i]
            S2 = data.iloc[:, j]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance_level:
                pairs.append((data.columns[i], data.columns[j]))
    return score_matrix, pvalue_matrix, pairs

scores, pvalues, pairs = find_cointegrated_pairs(data)
if not pairs:
    raise ValueError("No cointegrated pairs found")
print("Cointegrated pairs:", pairs)
```

### Step 3: Trading Algorithm

Generate trading signals based on the z-score of the spread between the prices of the cointegrated pairs.

```python
def zscore(series):
    return (series - series.mean()) / np.std(series)

pair = pairs[0]
S1 = data[pair[0]]
S2 = data[pair[1]]

spread = S1 - S2
zscore_spread = zscore(spread)

entry_threshold = 2.0
exit_threshold = 0.5

data['position'] = np.where(zscore_spread > entry_threshold, -1, 0)
data['position'] = np.where(zscore_spread < -entry_threshold, 1, data['position'])
data['position'] = np.where(np.abs(zscore_spread) < exit_threshold, 0, data['position'])
data['position'] = data['position'].shift(1)

data['returns'] = data[pair[0]].pct_change() - data[pair[1]].pct_change()
data['strategy_returns'] = data['position'] * data['returns']
data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()
```

### Step 4: Visualization

Visualize the results using the provided plotting functions.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the daily returns
plt.figure(figsize=(12, 6))
for asset in tickers:
    plt.plot(data.index, data[asset], label=asset)
plt.title('Daily Returns of Selected Assets')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.grid(True)
plt.show()

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data.pct_change().corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Selected Assets')
plt.show()

# Plot the cointegrated pairs
plt.figure(figsize=(12, 6))
for pair in pairs:
    asset1, asset2 = pair
    spread = data[asset1] - data[asset2]
    plt.plot(data.index, spread, label=f"{asset1}-{asset2}")
plt.title('Spread between Cointegrated Pairs')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.legend()
plt.grid(True)
plt.show()

# Plot cumulative returns
plt.figure(figsize=(12, 6))
data['cumulative_returns'].plot()
plt.title('Cumulative Returns of the Pairs Trading Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid(True)
plt.show()
```

## Interpreting Trading Signals

- **Entry Signal:** When the z-score of the spread exceeds the entry threshold (e.g., 2.0), a position is taken:
  - Short the spread (sell the first asset and buy the second asset) if the z-score is positive.
  - Long the spread (buy the first asset and sell the second asset) if the z-score is negative.

- **Exit Signal:** When the z-score of the spread falls below the exit threshold (e.g., 0.5), the position is closed.

## Adjusting Parameters

### Significance Level
The significance level in cointegration analysis determines the threshold for considering pairs as cointegrated. Lower significance levels (e.g., 0.05) are stricter, while higher levels (e.g., 0.10) are more lenient.

### Entry and Exit Thresholds
The entry and exit thresholds for the z-score determine when to open and close positions. These can be adjusted to optimize the performance of the strategy. Higher thresholds reduce the frequency of trades, while lower thresholds increase it.

### Optimization
The `optimize_strategy` function can be used to find the best combination of entry and exit thresholds that maximize the Sharpe ratio of the strategy.

```python
from itertools import product

def optimize_strategy(data, pair):
    best_sharpe = -np.inf
    best_params = None
    thresholds = np.arange(0.5, 3.5, 0.5)
    for entry_threshold, exit_threshold in product(thresholds, repeat=2):
        spread = data[pair[0]] - data[pair[1]]
        zscore_spread = zscore(spread)
        
        data['position'] = np.where(zscore_spread > entry_threshold, -1, 0)
        data['position'] = np.where(zscore_spread < -entry_threshold, 1, data['position'])
        data['position'] = np.where(np.abs(zscore_spread) < exit_threshold, 0, data['position'])
        data['position'] = data['position'].shift(1)
        
        data['strategy_returns'] = data['position'] * (data[pair[0]].pct_change() - data[pair[1]].pct_change())
        
        if data['strategy_returns'].isnull().any() or np.isinf(data['strategy_returns']).any():
            continue
        
        sharpe_ratio = data['strategy_returns'].mean() / data['strategy_returns'].std() * np.sqrt(252)
        
        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_params = (entry_threshold, exit_threshold)
    
    return best_params, best_sharpe

best_params, best_sharpe = optimize_strategy(data, pair)
print("Best parameters:", best_params)
print("Best Sharpe ratio:", best_sharpe)
```

## Conclusion

This user guide provides a comprehensive overview of how to use the pairs trading algorithm, interpret trading signals, and adjust parameters for optimal results. The strategy can be further refined and optimized based on additional research and testing.


## Acknowledgments

- Yahoo Finance for providing historical price data
- `statsmodels` library for cointegration tests
- `matplotlib` and `seaborn` for data visualization
