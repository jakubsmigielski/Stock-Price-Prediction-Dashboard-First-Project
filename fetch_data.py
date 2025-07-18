import yfinance as yf
import pandas as pd
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


symbols = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "GOOG": "Google"
}

start_date = "2020-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

for symbol, name in symbols.items():
    print(f" Downloading data for {symbol} ({name})...")
    df = yf.download(symbol, start=start_date, end=end_date)


    df.reset_index(inplace=True)
    df = df[["Date", "Close", "High", "Low", "Open", "Volume"]]


    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    df.to_csv(path, index=False, header=False)

    print(f" Saved to {path}")
