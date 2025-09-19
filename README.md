# Efficient-Frontier-Correlation-Heatmaps-Python
Simulates 100k portfolios (yfinance) to plot the efficient frontier, CAL, and correlation heatmap, ships a compact PDF report.

How to Use:
1. Configure
     - choose the tickers you want to use by changing this data frame (ETF, Stocks, crypto): df1 = pd.DataFrame(["BNP.PA", "MC.PA", "TTE.PA", "AIR.PA", "BN.PA"], columns=["Tickers"])
2. Set the analysis period
     - choose the time period (Line 16-17), 5 years is a good number to start
3. Risk-free rate (annual, decimal)
     - choose the right rf rate (line 18)
4. Risk aversion of the investor
     - will affect the indifference curve
5. Information for the pdf
     - the pdf will save in the working file
