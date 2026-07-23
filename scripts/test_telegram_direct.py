import urllib.request
import json

token = "8762045243:AAEBcdULLecPyINjb3xgwiBItRHBYAwQ1qg"
chat_id = "7808551739"
text = "🔔 TEST: BIST100 Yapay Zeka Telegram Bağlantı Testi (Direkt)"

url = f"https://api.telegram.org/bot{token}/sendMessage"
data = json.dumps({"chat_id": chat_id, "text": text}).encode('utf-8')

req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        res = response.read().decode('utf-8')
        print("TELEGRAM API RESPONSE:", res)
except Exception as e:
    print("TELEGRAM API ERROR:", e)
