import ipywidgets as widgets
import pandas as pd
from IPython.display import display

tickers = ["MSFT", "AAPL", "GOOGL", "NVDA", "PG"] 

df1 = pd.DataFrame(tickers, columns=["Tickers"])


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
if df1 is None: df1 = pd.DataFrame(["MSFT","AAPL","GOOGL","NVDA","UL"], columns=["Tickers"])
tickers = df1["Tickers"].tolist()


for ticker in tickers:
    ticker = yf.Ticker(ticker)
    ticker.history(start = '2019-12-1', end = '2024-12-1')
    ticker_daily_return = ticker.history(start = '2019-12-1', end = '2024-12-1')
    ticker_monthly_return = ticker_daily_return.resample("1ME").last()
    ticker_monthly_return = ticker_monthly_return["Close"].diff()/ticker_monthly_return["Close"]
    def winsorized(data, limits=[0.05, 0.05]):
      return winsorize(ticker_monthly_return, limits=limits)
      ticker_monthly_return = ticker_monthly_return.apply(winsorized)
      ticker_expected_return = ticker_monthly_return.mean()
      ticker_expected_return = ticker_expected_return * 12
      ticker_std = ticker_monthly_return.std()
      ticker_std = ticker_std * np.sqrt(12)
    print(ticker)
    print("\033[1m""\033[94m" + 'monthly returns' + "\033[0m""\033[0m")
    print(ticker_monthly_return)

import yfinance as yf
import pandas as pd

##tickers = ["MSFT", "AAPL", "GOOGL", "NVDA", "UL"]

data = yf.download(tickers, start="2019-12-01", end="2024-12-01")#"["Adj Close"]

data = data.resample("M").last()

data = data.dropna()

returns = (data.diff() / data).dropna()

cov_matrix = returns.cov() * 12


print("\nCovariance Matrix of Returns:")
print(cov_matrix)

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# %matplotlib inline

good_stocks = tickers #["MSFT", "AAPL", "GOOGL", "NVDA", "UL"]
count = len(good_stocks)

symbols = []
for ticker in tickers:
    tick = yf.Ticker(ticker)
    good_history = tick.history(period='5y')
    good_history.index = pd.to_datetime(good_history.index)
    good_history['Symbol'] = ticker
    symbols.append(good_history)

good_df = pd.concat(symbols)
good_df = good_df[['Close', 'Symbol']].reset_index()

price = good_df.pivot(index='Date', columns='Symbol', values='Close')

month_price = price.resample("M").last()
month_ret = month_price.pct_change().dropna()

mean = month_ret.mean() * 12
std = month_ret.std() * np.sqrt(12)
cov = month_ret.cov()
if month_ret.empty:
    raise ValueError("Erreur : 'month_ret' est vide. Vérifiez les données récupérées.")
p_ret = []
p_vol = []
p_weights = []
num_portfolios = 100000

np.random.seed(42)

for portfolio in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(count))
    weights = np.clip(weights, 0.0, 0.30)  # max 30% par actif
    weights /= weights.sum() 
    returns = np.dot(weights, mean)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)) * 12)
    p_ret.append(returns)
    p_vol.append(vol)
    p_weights.append(weights)

portfolios = pd.DataFrame({'Returns': p_ret, 'Volatility': p_vol})
for i, symbol in enumerate(good_stocks):
    portfolios[f"{symbol} Weight"] = [w[i] for w in p_weights]

rf = 0.02
portfolios['Sharpe Ratio'] = (portfolios['Returns'] - rf) / portfolios['Volatility']

good_optimal_risky_port = portfolios.iloc[portfolios['Sharpe Ratio'].idxmax()]
print(f"OPTIMAL RISKY PORTFOLIO\n")
print(good_optimal_risky_port)
good_min_var_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
print(f"MINIMUM VARIANCE PORTFOLIO\n")
print(good_min_var_port)

def check_sum(weights):
    return np.sum(weights) - 1

def minimize_volatility(weights, mean, cov):
    _, vol = get_ret_vol(weights, mean, cov)
    return vol

def get_ret_vol(weights, mean, cov):
    ret = np.dot(weights, mean)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)) * 12)
    return ret, vol

def efficient_frontier(mean, cov, rf):
    good_frontier_x = []
    good_frontier_y = []
    for target_return in np.linspace(good_min_var_port['Returns'], max(portfolios['Returns']), 100):
        cons = [{'type': 'eq', 'fun': check_sum},
                {'type': 'eq', 'fun': lambda w: get_ret_vol(w, mean, cov)[0] - target_return}]
        result = minimize(lambda w: minimize_volatility(w, mean, cov), [1/count] * count, bounds=[(0, 1)] * count, constraints=cons)
        if result.success:
            good_frontier_x.append(result.fun)
            good_frontier_y.append(target_return)
    return good_frontier_x, good_frontier_y

good_frontier_x, good_frontier_y = efficient_frontier(mean, cov, rf)

risk_free_CAL = 0.02
good_cal_x = np.linspace(0, max(good_frontier_x)*1.5, 1000)
cal_slope = (good_optimal_risky_port['Returns'] - risk_free_CAL) / good_optimal_risky_port['Volatility']
good_cal_y = risk_free_CAL + cal_slope * good_cal_x

rf = 0.02
A = 4

good_Rp = good_optimal_risky_port['Returns']
good_sigma_p = good_optimal_risky_port['Volatility']

y = (good_Rp - rf) / (A * good_sigma_p**2)

Rc = y * good_Rp + (1 - y) * rf
good_sigma_c = y * good_sigma_p

U = Rc - 0.5 * A * good_sigma_c**2

good_indifference_sigma = np.linspace(0, good_sigma_c * 1.5, 100)
good_indifference_curve = U + 0.5 * A * good_indifference_sigma**2

print(f"Complete Optimal Portfolio Return (Rc): {Rc:.2%}")
print(f"Complete Optimal Portfolio Volatility (good_sigma_c): {good_sigma_c:.2%}")
print(f"Proportion Invested in Risky Portfolio (y): {y:.2%}")

plt.figure(figsize=(20, 12))
portfolios.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio', s = 2, cmap='viridis', grid=True, colorbar=True)

plt.plot(good_frontier_x, good_frontier_y, 'r--', label='Efficient Frontier')

plt.scatter(good_optimal_risky_port['Volatility'], good_optimal_risky_port['Returns'], color='blue', s=50, label='Optimal Risky Portfolio')
plt.scatter(good_min_var_port['Volatility'], good_min_var_port['Returns'], color='orange', s=50, label='Minimum Variance Portfolio')

plt.plot(good_cal_x, good_cal_y, label="Capital Allocation Line (CAL)", color='red', linewidth=2)

plt.plot(good_indifference_sigma, good_indifference_curve, label="Indifference Curve", linestyle="--", color="green", linewidth=2)

plt.scatter(good_sigma_c, Rc, color='purple', s=100, label='Complete Optimal Portfolio', edgecolors='black')

plt.title("Efficient Frontier with CAL and Indifference Curve")
plt.xlabel("Volatility")
plt.ylabel("Expected Returns")
plt.legend()
plt.savefig("efficientfrontCAL.png")
plt.grid(True)
plt.show()

"""# Partie Y

Step 1 : packages
"""

# The packages that we're going to use

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns



# Setting the dataframe

start = datetime (2015,1,1)
end = datetime (2020,1,1)
now = datetime.now()

# Test if we get the correct datetime

end - start

# Setting the tickers that we want to study

good_stocks = tickers
'''
MSFT. = Microsoft
AAPL. = Apple
GOOGL. = Google
NVDA. = Nvidia
UL. = Unilever
'''

table = pd.DataFrame ()

for ticker in good_stocks:
  data = yf.download (ticker, start, end = now)
  table[ticker] = data["Close"]

stocks_table = pd.concat ([table.dropna()])
len(stocks_table)

stocks_table.pct_change()

stocks_table.pct_change().corr()

sns.heatmap(stocks_table.pct_change().corr(), cmap="YlGnBu", annot = True)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.heatmap(stocks_table.pct_change().corr(), cmap="YlGnBu", annot = True)
plt.title("Matrix of correlation")
plt.savefig("correlation_matrix.png")
plt.close()

from fpdf import FPDF

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

pdf.set_font("Arial", size=16)
pdf.cell(200, 10, txt="Report of Correlation analysis", ln=True, align='C')

pdf.set_font("Arial", size=12)
pdf.ln(10)
pdf.multi_cell(0, 10, txt="Heatmap of correlation")

pdf.ln(10)
pdf.image("correlation_matrix.png", x=10, y=None, w=180)
pdf.add_page()
pdf.set_font("Arial", size=16)
pdf.cell(200, 10, txt="Efficient Frontier with CAL and Indifference Curve", ln=True, align='C')
pdf.ln(10)
pdf.image("efficientfrontCAL.png", x=10, y=None, w=180)
def footer(self):
        # Position cursor at 1.5 cm from bottom:
  self.set_y(-15)
        # Setting font: helvetica italic 8
  self.set_font("helvetica", style="I", size=8)
        # Printing page number:
  self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")
pdf.output("report_correlation.pdf")
print("PDF saved as report_correlation.pdf — open it from your working directory.")
