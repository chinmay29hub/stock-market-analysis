import yfinance as yf
import pandas as pd

data_df = yf.download("GOOG", start="2021-02-01", end="2021-03-01")
print(data_df)
data_df.to_csv('Googlelivedata.csv')
