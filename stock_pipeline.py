# -*- coding: utf-8 -*-
"""
PIPELINE PRIX SEULEMENT - yfinance + Tiingo → Drive
AUTOMATIQUE
"""

import os
import pandas as pd
from datetime import datetime
import yfinance as yf
from tiingo import TiingoClient
import subprocess  # ← Pour git commit/push



# ====================== CONFIG ======================
TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY manquante !")

DATA_FOLDER = "data"  # ← Dossier dans ton repo
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(f"{DATA_FOLDER}/tiingo", exist_ok=True)

def collect_yfinance():
    symbols = ["AAPL", "TSLA", "MSFT", "BTC-USD", "GOOGL"]
    path = "/tmp/stock_data"
    os.makedirs(path, exist_ok=True)
    all_data = []
    print("yfinance...")
    for s in symbols:
        df = yf.Ticker(s).history(period="1y").reset_index()
        df['symbol'] = s
        df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
        all_data.append(df)
        df.to_csv(f"{path}/{s}.csv", index=False)
    pd.concat(all_data).to_csv(f"{path}/ALL_YFINANCE.csv", index=False)
    return path

def collect_tiingo():
    client = TiingoClient({'api_key': TIINGO_API_KEY, 'session': True})
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL"]
    path = "/tmp/stock_data/tiingo"
    os.makedirs(path, exist_ok=True)
    all_data = []
    print("Tiingo prix...")
    for s in symbols:
        df = client.get_dataframe(s, frequency='daily', startDate=datetime.now().replace(year=datetime.now().year-1))
        df = df.reset_index()
        df['symbol'] = s
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
        df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']
        all_data.append(df)
        df.to_csv(f"{path}/{s}_prices.csv", index=False)
    pd.concat(all_data).to_csv(f"{path}/ALL_TIINGO.csv", index=False)
    return "/tmp/stock_data"

# ====================== GIT COMMIT + PUSH ======================
def commit_and_push():
    print("Commit + Push sur GitHub...")
    subprocess.run(["git", "config", "--global", "user.adam.hassen@esprit.tn", "actions@github.com"])
    subprocess.run(["git", "config", "--global", "user.adam", "GitHub Actions"])
    subprocess.run(["git", "add", DATA_FOLDER])
    subprocess.run(["git", "commit", "-m", f"Update prix {datetime.now().strftime('%Y-%m-%d')}"])
    subprocess.run(["git", "push"])
    print("TOUT PUSHÉ SUR GIT !")

# ====================== MAIN ======================
if __name__ == "__main__":
    print("DÉBUT PIPELINE GIT -", datetime.now().strftime("%Y-%m-%d %H:%M"))
    collect_yfinance()
    collect_tiingo()
    commit_and_push()
    print("TERMINÉ – PRIX DANS LE REPO GIT")
