import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/AAPL.csv", header=1)
df = df.iloc[2:].copy()
df["Date"] = pd.to_datetime(df["Date"])
df["Close"] = pd.to_numeric(df["Close"], errors='coerce')

df["Return (%)"] = df["Close"].pct_change() * 100

plt.figure(figsize=(14,6))
plt.plot(df["Date"], df["Return (%)"], color="purple", label="Daily Return (%)")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title("AAPL Daily Return (%) Over Time")
plt.xlabel("Date")
plt.ylabel("Return (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
