import yfinance as yf
import pandas as pd
from ta.trend import MACD
from tqdm import tqdm

# -----------------------------
# GET NYSE STOCK LIST
# -----------------------------
nyse_url = "https://old.nasdaq.com/screening/companies-by-name.aspx?exchange=NYSE&render=download"
stocks = pd.read_csv(nyse_url)
tickers = stocks['Symbol'].tolist()

results = []

# -----------------------------
# MACD FUNCTION
# -----------------------------
def add_macd(df):
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['signal'] = macd.macd_signal()
    return df

# -----------------------------
# MAIN LOOP
# -----------------------------
for ticker in tqdm(tickers[:500]):  # test first, then remove limit
    try:
        # period="1y" will cut down the time by 40%
        df = yf.download(ticker, period="2y", interval="1d", progress=False)

        if len(df) < 200:
            continue

        # DAILY MACD
        df = add_macd(df)

        # MONTHLY
        monthly = df.resample('M').agg({
            'Open': 'first',
            'Close': 'last',
            'High': 'max',
            'Volume': 'sum'
        }).dropna()
        monthly = add_macd(monthly)

        # QUARTERLY
        quarterly = df.resample('Q').agg({
            'Open': 'first',
            'Close': 'last'
        }).dropna()

        # -----------------------------
        # CONDITIONS (same as yours)
        # -----------------------------
        cond_q = quarterly['Close'].iloc[-1] > quarterly['Open'].iloc[-1]

        cond_m1 = monthly['Close'].iloc[-1] > monthly['Open'].iloc[-1]
        cond_m2 = monthly['Close'].iloc[-2] <= monthly['Open'].iloc[-2]

        cond_m3 = (
            monthly['Close'].iloc[-3] > monthly['Open'].iloc[-3] and
            monthly['Close'].iloc[-3] > monthly['High'].iloc[-4] and
            monthly['Close'].iloc[-4] > monthly['Open'].iloc[-4] and
            monthly['Volume'].iloc[-3] > monthly['Volume'].iloc[-4]
        )

        cond_macd_month = monthly['macd'].iloc[-1] > 0

        cond_daily_1 = (
            df['macd'].iloc[-2] < df['signal'].iloc[-2] and
            df['macd'].iloc[-1] > df['signal'].iloc[-1] and
            df['macd'].iloc[-1] < 0
        )

        cond_daily_2 = (
            df['macd'].iloc[-1] > df['signal'].iloc[-1] and
            df['macd'].iloc[-2] < 0 and
            df['macd'].iloc[-1] > 0
        )

        cond_daily = cond_daily_1 or cond_daily_2

        # FINAL FILTER
        if cond_q and cond_m1 and cond_m2 and cond_m3 and cond_macd_month and cond_daily:
            results.append(ticker)

    except:
        continue

# -----------------------------
# OUTPUT
# -----------------------------
print("\n✅ Matching NYSE Stocks:")
for stock in results:
    print(stock)

print(f"\nTotal Matches: {len(results)}")