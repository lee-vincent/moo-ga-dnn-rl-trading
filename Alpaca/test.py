import config
import requests

BASE_URL = "https://paper-api.alpaca.markets/v2/"

HEADERS = {
    "accept": "application/json",
    "APCA-API-KEY-ID": config.ALPACA_KEY,
    "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY
}

# response = requests.get(BASE_URL + "account", headers=HEADERS)
# print(response.json(), '\n')

payload = {
  "side": "buy",
  "type": "market",
  "time_in_force": "day",
  "qty": "5",
  "symbol": "TQQQ",
}

response = requests.post(BASE_URL + "orders", json=payload, headers=HEADERS)
print("MAKE ORDER", response.text)

response = requests.get(BASE_URL + "orders", headers=HEADERS)
print("GET ORDERS", response.text)
