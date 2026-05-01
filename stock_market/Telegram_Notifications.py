import yfinance as yf
import pandas as pd
import requests
from ta.trend import MACD
from tqdm import tqdm

# -----------------------------
# TELEGRAM CONFIG
# -----------------------------
BOT_TOKEN = "8672570128:AAFjIElNYcTcOMlomUdKmZ-8JPZTwii1RNo"
CHAT_ID = "8762329875"

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=payload)

# -----------------------------
# GET NASDAQ STOCKS
# -----------------------------
nasdaq_url = "https://old.nasdaq.com/screening/companies-by-name.aspx?exchange=NASDAQ&render=download"
stocks = pd.read_csv(nasdaq_url)
tickers = stocks['Symbol'].tolist()

# -----------------------------
# MACD FUNCTION
# -----------------------------
def add_macd(df):
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['signal'] = macd.macd_signal()
    return df

# -----------------------------
# SCANNER
# -----------------------------
results = []

for ticker in tqdm(tickers[:500]):  # remove limit later
    try:
        # period="1y" will cut down the time by 40%
        df = yf.download(ticker, period="2y", interval="1d", progress=False)

        if len(df) < 200:
            continue

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

        # CONDITIONS
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

        # FINAL
        if cond_q and cond_m1 and cond_m2 and cond_m3 and cond_macd_month and cond_daily:
            results.append(ticker)
            send_telegram(f"🔥 Stock Match: {ticker}")

    except:
        continue

# FINAL SUMMARY MESSAGE
send_telegram(f"✅ Scan Complete. Total Matches: {len(results)}\n{results}")