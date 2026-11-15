import os
import time
import datetime
import requests
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 12 Core Global Multi-Asset Universe + 2 Macro Benchmarks
MULTI_ASSETS = {
    # Equities
    "SPY": "S&P 500 US Large Cap",
    "QQQ": "Nasdaq 100 Tech",
    "EEM": "MSCI Emerging Markets",
    # Fixed Income
    "IEF": "7-10 Year US Treasury",
    "TLT": "20+ Year US Treasury",
    "HYG": "High Yield Corporate Bond",
    "LQD": "Investment Grade Corporate Bond",
    # Real Assets & Commodities
    "GLD": "SPDR Gold Shares",
    "DBC": "Invesco DB Commodity Index",
    "VNQ": "Vanguard Real Estate (REITs)",
    # Currencies & Cash
    "UUP": "Invesco DB US Dollar Bullish",
    "SHV": "iShares Short Treasury (Cash Proxy)",
    # Alternative / Crypto
    "BTC-USD": "Bitcoin (Crypto Asset)"
}

BENCHMARKS = ["^VIX", "^IRX"]
ALL_TICKERS = list(MULTI_ASSETS.keys()) + BENCHMARKS

start_dt = datetime.datetime(2018, 1, 1)
end_dt = datetime.datetime(2026, 2, 1)
start_ts = int(start_dt.timestamp())
end_ts = int(end_dt.timestamp())

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

daily_series = {}

print(f"[INFO] Fetching daily historical data from Yahoo Finance API ({start_dt.date()} to {end_dt.date()})...")

for ticker in ALL_TICKERS:
    # URL encode ticker if needed (e.g., ^VIX -> %5EVIX, but query1 accepts directly or encoded)
    encoded_ticker = requests.utils.quote(ticker)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_ticker}?period1={start_ts}&period2={end_ts}&interval=1d"
    success = False
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                result = data["chart"]["result"][0]
                timestamps = result["timestamp"]
                
                # Check if adjclose exists
                if "adjclose" in result.get("indicators", {}) and result["indicators"]["adjclose"] and "adjclose" in result["indicators"]["adjclose"][0]:
                    prices = result["indicators"]["adjclose"][0]["adjclose"]
                else:
                    prices = result["indicators"]["quote"][0]["close"]
                    
                dates = [datetime.datetime.utcfromtimestamp(ts).date() for ts in timestamps]
                s = pd.Series(prices, index=pd.to_datetime(dates), name=ticker)
                daily_series[ticker] = s
                print(f"  [+] {ticker:8s}: {len(s)} daily rows fetched.")
                success = True
                break
            else:
                time.sleep(1)
        except Exception as e:
            time.sleep(1)
    if not success:
        print(f"  [-] {ticker:8s}: FAILED after 3 attempts.")
    time.sleep(0.3)

# Combine into single DataFrame
df_daily = pd.DataFrame(daily_series)
df_daily = df_daily.sort_index()
df_daily = df_daily.ffill().bfill()

daily_csv = os.path.join(DATA_DIR, "multiasset_daily_prices.csv")
df_daily.to_csv(daily_csv)
print(f"\n[SUCCESS] Daily prices saved to: {daily_csv} (Shape: {df_daily.shape})")

# Calculate Daily Returns
df_daily_returns = df_daily.pct_change().dropna()
daily_ret_csv = os.path.join(DATA_DIR, "multiasset_daily_returns.csv")
df_daily_returns.to_csv(daily_ret_csv)

# Monthly Resampling (Month-End)
try:
    df_monthly_prices = df_daily.resample("ME").last()
except Exception:
    df_monthly_prices = df_daily.resample("M").last()

# Calculate Monthly Returns
df_monthly_returns = df_monthly_prices.pct_change().dropna()

# Asset monthly returns
asset_cols = [c for c in MULTI_ASSETS.keys() if c in df_monthly_returns.columns]
df_asset_returns = df_monthly_returns[asset_cols]
monthly_csv = os.path.join(DATA_DIR, "multiasset_monthly_returns.csv")
df_asset_returns.to_csv(monthly_csv, float_format="%.6f")
print(f"[SUCCESS] Multi-Asset monthly returns saved to: {monthly_csv} (Shape: {df_asset_returns.shape})")

# Macro levels & returns
macro_cols = [c for c in BENCHMARKS if c in df_monthly_prices.columns]
df_macro_prices = df_monthly_prices[macro_cols]
macro_levels_csv = os.path.join(DATA_DIR, "macro_monthly_levels.csv")
df_macro_prices.to_csv(macro_levels_csv, float_format="%.4f")

macro_returns_csv = os.path.join(DATA_DIR, "macro_monthly_returns.csv")
df_monthly_returns[macro_cols].to_csv(macro_returns_csv, float_format="%.6f")
print(f"[SUCCESS] Macro data saved to: {macro_levels_csv} & {macro_returns_csv}")

print("\n--- Summary Statistics of Multi-Asset Universe (Annualized) ---")
summary_stats = pd.DataFrame({
    "Asset Class": [MULTI_ASSETS[s] for s in df_asset_returns.columns],
    "Mean Return (%)": df_asset_returns.mean() * 12 * 100,
    "Volatility (%)": df_asset_returns.std() * np.sqrt(12) * 100,
    "Downside Vol (%)": df_asset_returns.apply(lambda col: np.sqrt(12) * np.std(np.minimum(col - 0.0575/12, 0))) * 100,
    "Sharpe Ratio": (df_asset_returns.mean() * 12) / (df_asset_returns.std() * np.sqrt(12)),
    "Min Month (%)": df_asset_returns.min() * 100,
    "Max Month (%)": df_asset_returns.max() * 100
})
summary_csv = os.path.join(DATA_DIR, "multiasset_summary_statistics.csv")
summary_stats.to_csv(summary_csv)
print(summary_stats.to_string())
