import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/AAPL.csv", skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"])

df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

df.set_index("Date", inplace=True)

df["Next_Close"] = df["Close"].shift(-1)
df.dropna(inplace=True)

X = df[["Close"]]
y = df["Next_Close"]
model = LinearRegression()
model.fit(X, y)

df["Predicted_Close"] = model.predict(X)

plt.figure(figsize=(14, 7))
plt.plot(df.index, df["Close"], label="Actual Close")
plt.plot(df.index, df["Predicted_Close"], label="Predicted Close", linestyle="--")
plt.title("AAPL Price Prediction using Linear Regression")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




