import yfinance as yf
import pandas as pd
from ta.trend import MACD
from tqdm import tqdm

# -----------------------------
# Get NASDAQ stock list
# -----------------------------
nasdaq_url = "https://old.nasdaq.com/screening/companies-by-name.aspx?exchange=NASDAQ&render=download"
stocks = pd.read_csv(nasdaq_url)
tickers = stocks['Symbol'].tolist()

results = []

# -----------------------------
# Helper function: MACD
# -----------------------------
def add_macd(df):
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['signal'] = macd.macd_signal()
    return df

# -----------------------------
# Main Loop
# -----------------------------
for ticker in tqdm(tickers[:500]):  # limit for speed (remove slice for full scan)
    try:
        # period="1y" will cut down the time by 40%
        df = yf.download(ticker, period="2y", interval="1d", progress=False)

        if len(df) < 200:
            continue

        # DAILY MACD
        df = add_macd(df)

        # MONTHLY DATA
        monthly = df.resample('M').agg({
            'Open': 'first',
            'Close': 'last',
            'High': 'max',
            'Volume': 'sum'
        }).dropna()

        monthly = add_macd(monthly)

        # QUARTERLY DATA
        quarterly = df.resample('Q').agg({
            'Open': 'first',
            'Close': 'last'
        }).dropna()

        # -----------------------------
        # CONDITIONS
        # -----------------------------

        # Quarterly
        cond_q = quarterly['Close'].iloc[-1] > quarterly['Open'].iloc[-1]

        # Monthly conditions
        cond_m1 = monthly['Close'].iloc[-1] > monthly['Open'].iloc[-1]
        cond_m2 = monthly['Close'].iloc[-2] <= monthly['Open'].iloc[-2]

        cond_m3 = (
            monthly['Close'].iloc[-3] > monthly['Open'].iloc[-3] and
            monthly['Close'].iloc[-3] > monthly['High'].iloc[-4] and
            monthly['Close'].iloc[-4] > monthly['Open'].iloc[-4] and
            monthly['Volume'].iloc[-3] > monthly['Volume'].iloc[-4]
        )

        # Monthly MACD > 0
        cond_macd_month = monthly['macd'].iloc[-1] > 0

        # Daily MACD conditions
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

        # FINAL CONDITION
        if cond_q and cond_m1 and cond_m2 and cond_m3 and cond_macd_month and cond_daily:
            results.append(ticker)

    except Exception as e:
        continue

# -----------------------------
# OUTPUT
# -----------------------------
print("\nMatching Stocks:")
print(results)