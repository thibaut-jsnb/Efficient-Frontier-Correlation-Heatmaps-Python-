# Efficient-Frontier-Correlation-Heatmaps-Python
Simulates 100k portfolios (yfinance) to plot the efficient frontier, CAL, and correlation heatmap, ships a compact PDF report.

How to Use:
1. Configure
     - choose the tickers you want to use by changing this (ETF, Stocks, crypto): tickers = ["MSFT", "AAPL", "GOOGL", "NVDA", "PG"]

2. Set the analysis period
     - choose the time period (Line 23-24), 5 years is a good number to start
3. Risk-free rate (annual, decimal)
     - choose the right rf rate (line 109)
4. Risk aversion of the investor
     - will affect the indifference curve
5. Information for the pdf
     - the pdf will save in the working file
