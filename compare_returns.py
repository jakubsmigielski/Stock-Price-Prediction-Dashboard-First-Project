import pandas as pd
import matplotlib.pyplot as plt

# Wczytaj dane
aapl = pd.read_csv("data/AAPL.csv", header=1).iloc[2:].copy()
tsla = pd.read_csv("data/TSLA.csv", header=1).iloc[2:].copy()
goog = pd.read_csv("data/GOOG.csv", header=1).iloc[2:].copy()

# Konwersja daty i zamknięcia
for df in [aapl, tsla, goog]:
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
    df["Return (%)"] = df["Close"].pct_change() * 100

# Wykres
plt.figure(figsize=(14, 6))
plt.plot(aapl["Date"], aapl["Return (%)"], label="Apple (AAPL)", alpha=0.8)
plt.plot(tsla["Date"], tsla["Return (%)"], label="Tesla (TSLA)", alpha=0.8)
plt.plot(goog["Date"], goog["Return (%)"], label="Google (GOOG)", alpha=0.8)

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title("Daily Stock Returns (%) - AAPL vs TSLA vs GOOG")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
