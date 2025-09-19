import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from scipy.optimize import minimize
from fpdf import FPDF
from datetime import datetime

# ----------------------
# Parameters / Tickers
# ----------------------
df1 = pd.DataFrame(["BNP.PA", "MC.PA", "TTE.PA", "AIR.PA", "BN.PA"], columns=["Tickers"])
tickers = df1["Tickers"].tolist()
start_prices = "2019-12-01"
end_prices   = "2024-12-01"
risk_free    = 0.02
risk_aversion_A = 4

# ----------------------
# 1) Per-ticker monthly returns (winsorized 5% tails)
# ----------------------
for tkr in tickers:
    tk = yf.Ticker(tkr)
    h = tk.history(start=start_prices, end=end_prices)
    if h.empty:
        raise ValueError(f"No data for {tkr}")
    mpx = h["Close"].resample("M").last()
    mret = mpx.pct_change().dropna()
    wret = pd.Series(winsorize(mret, limits=[0.05, 0.05]), index=mret.index)
    print("\n", tkr, "— monthly returns (winsorized):")
    print(wret)

# ----------------------
# 2) Covariance matrix (annualized) from monthly returns
# ----------------------
bulk = yf.download(tickers, start=start_prices, end=end_prices, auto_adjust=True, progress=False)
if isinstance(bulk.columns, pd.MultiIndex):
    prices = bulk["Adj Close"] if "Adj Close" in bulk.columns.get_level_values(0) else bulk["Close"]
else:
    prices = bulk
prices = prices.resample("M").last().dropna(how="all")
returns = prices.pct_change().dropna()
cov_matrix = returns.cov() * 12.0
print("\nCovariance matrix (annualized):")
print(cov_matrix)

# ----------------------
# 3) Efficient frontier (random portfolios + min-vol frontier)
# ----------------------
good_stocks = list(prices.columns)
n = len(good_stocks)
mean = returns.mean() * 12.0
cov  = returns.cov() * 12.0

np.random.seed(42)
num_portfolios = 100000
p_ret, p_vol, p_w = [], [], []

for _ in range(num_portfolios):
    w = np.random.dirichlet(np.ones(n))
    r = float(np.dot(w, mean))
    v = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
    p_ret.append(r); p_vol.append(v); p_w.append(w)

portfolios = pd.DataFrame({"Returns": p_ret, "Volatility": p_vol})
for i, s in enumerate(good_stocks):
    portfolios[f"{s} Weight"] = [w[i] for w in p_w]

portfolios["Sharpe Ratio"] = (portfolios["Returns"] - risk_free) / portfolios["Volatility"]
opt_risky = portfolios.iloc[portfolios["Sharpe Ratio"].idxmax()]
min_var   = portfolios.iloc[portfolios["Volatility"].idxmin()]
print("\nOptimal risky portfolio:"); print(opt_risky)
print("\nMinimum variance portfolio:"); print(min_var)

def check_sum(w: np.ndarray) -> float:
    return np.sum(w) - 1.0

def get_ret_vol(w: np.ndarray, mu: pd.Series, c: pd.DataFrame):
    r = float(np.dot(w, mu))
    v = float(np.sqrt(np.dot(w.T, np.dot(c, w))))
    return r, v

def minimize_vol(w: np.ndarray, mu: pd.Series, c: pd.DataFrame):
    return get_ret_vol(w, mu, c)[1]

def efficient_frontier(mu: pd.Series, c: pd.DataFrame):
    xs, ys = [], []
    grid = np.linspace(float(min_var["Returns"]), float(portfolios["Returns"].max()), 100)
    for tr in grid:
        cons = ({"type": "eq", "fun": check_sum},
                {"type": "eq", "fun": lambda w, t=tr: get_ret_vol(w, mu, c)[0] - t})
        res = minimize(lambda w: minimize_vol(w, mu, c), x0=np.ones(n)/n,
                       bounds=[(0.0, 1.0)]*n, constraints=cons, method="SLSQP",
                       options={"maxiter": 1000, "ftol": 1e-9})
        if res.success:
            xs.append(res.fun); ys.append(tr)
    return np.array(xs), np.array(ys)

front_x, front_y = efficient_frontier(mean, cov)

# CAL and complete portfolio (utility with A)
cal_x = np.linspace(0, max(front_x)*1.5, 1000)
cal_slope = (opt_risky["Returns"] - risk_free) / opt_risky["Volatility"]
cal_y = risk_free + cal_slope * cal_x
Rp, sigp = float(opt_risky["Returns"]), float(opt_risky["Volatility"])
y_star = (Rp - risk_free) / (risk_aversion_A * sigp**2)
Rc = y_star * Rp + (1 - y_star) * risk_free
sigc = y_star * sigp
U = Rc - 0.5 * risk_aversion_A * sigc**2
sigma_indiff = np.linspace(0, sigc*1.5, 100)
indiff = U + 0.5 * risk_aversion_A * sigma_indiff**2
print(f"\nComplete portfolio Rc: {Rc:.2%} | sigma_c: {sigc:.2%} | y*: {y_star:.2%}")

# ----------------------
# 4) Plots
# ----------------------
plt.figure(figsize=(20, 12))
portfolios.plot.scatter(x="Volatility", y="Returns", c="Sharpe Ratio", s=2, cmap="viridis", grid=True, colorbar=True)
plt.plot(front_x, front_y, "r--", label="Efficient Frontier")
plt.scatter(opt_risky["Volatility"], opt_risky["Returns"], s=60, label="Optimal Risky Portfolio")
plt.scatter(min_var["Volatility"], min_var["Returns"], s=60, label="Minimum Variance Portfolio")
plt.plot(cal_x, cal_y, label="Capital Allocation Line (CAL)", linewidth=2)
plt.plot(sigma_indiff, indiff, label="Indifference Curve", linestyle="--", linewidth=2)
plt.scatter(sigc, Rc, s=120, label="Complete Portfolio", edgecolors="black")
plt.title("Efficient Frontier with CAL and Indifference Curve")
plt.xlabel("Volatility (annualized)"); plt.ylabel("Expected Return (annualized)")
plt.legend(); plt.grid(True)
plt.savefig("efficientfrontCAL.png", bbox_inches="tight", dpi=150)
plt.close()

# Correlation heatmap
start_corr = datetime(2015, 1, 1); now = datetime.now()
tbl = pd.DataFrame()
for t in good_stocks:
    d = yf.download(t, start=start_corr, end=now, progress=False, auto_adjust=True)
    tbl[t] = d["Close"]
corr = tbl.dropna().pct_change().corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="YlGnBu", annot=True, fmt=".2f", square=True)
plt.title("Correlation Matrix (Close Returns)")
plt.tight_layout()
plt.savefig("correlation_matrix.png", bbox_inches="tight", dpi=150)
plt.close()

# ----------------------
# 5) PDF report (compiles the two figures)
# ----------------------
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=16)
pdf.cell(0, 10, "Report of Correlation Analysis", ln=True, align="C")
pdf.set_font("Arial", size=12); pdf.ln(6)
pdf.multi_cell(0, 8, "Heatmap of correlations between selected assets (close returns).")
pdf.ln(6); pdf.image("correlation_matrix.png", x=15, y=None, w=180)
pdf.add_page(); pdf.set_font("Arial", size=16)
pdf.cell(0, 10, "Efficient Frontier with CAL and Indifference Curve", ln=True, align="C")
pdf.ln(6); pdf.image("efficientfrontCAL.png", x=15, y=None, w=180)
pdf.output("report_correlation.pdf")
print("Saved: efficientfrontCAL.png, correlation_matrix.png, report_correlation.pdf")