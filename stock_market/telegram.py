import requests

BOT_TOKEN = "8672570128:AAFjIElNYcTcOMlomUdKmZ-8JPZTwii1RNo"
CHAT_ID = "8762329875"

requests.post(
    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
    data={"chat_id": CHAT_ID, "text": "✅ Bot working!"}
)