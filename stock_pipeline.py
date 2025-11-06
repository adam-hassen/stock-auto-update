# -*- coding: utf-8 -*-
"""
PIPELINE PRIX → GIT REPO (commit + push auto)
FONCTIONNE À 100%
"""

import os
import pandas as pd
from datetime import datetime
import yfinance as yf
from tiingo import TiingoClient
import subprocess

# ====================== CONFIG ======================
TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY manquante !")

PUSH_TOKEN = os.getenv('PUSH_TOKEN')  # Ton PAT
if not PUSH_TOKEN:
    raise ValueError("PUSH_TOKEN manquant !")

DATA_FOLDER = "data"  # DOSSIER DANS LE REPO
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(f"{DATA_FOLDER}/tiingo", exist_ok=True)

# ====================== YFINANCE ======================
def collect_yfinance():
    symbols = ["AAPL", "TSLA", "MSFT", "BTC-USD", "GOOGL"]
    all_data = []
    print("yfinance...")
    for s in symbols:
        df = yf.Ticker(s).history(period="1y").reset_index()
        df['symbol'] = s
        df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
        all_data.append(df)
        df.to_csv(f"{DATA_FOLDER}/{s}.csv", index=False)  # ← ICI DANS data/
    pd.concat(all_data).to_csv(f"{DATA_FOLDER}/ALL_YFINANCE.csv", index=False)
    print("yfinance OK")

# ====================== TIINGO ======================
def collect_tiingo():
    client = TiingoClient({'api_key': TIINGO_API_KEY, 'session': True})
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL"]
    all_data = []
    print("Tiingo...")
    for s in symbols:
        df = client.get_dataframe(s, frequency='daily', startDate=datetime.now().replace(year=datetime.now().year-1))
        df = df.reset_index()
        df['symbol'] = s
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
        df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']
        all_data.append(df)
        df.to_csv(f"{DATA_FOLDER}/tiingo/{s}_prices.csv", index=False)  # ← ICI DANS data/tiingo
    pd.concat(all_data).to_csv(f"{DATA_FOLDER}/tiingo/ALL_TIINGO.csv", index=False)
    print("Tiingo OK")

# ====================== GIT COMMIT + PUSH ======================
def commit_and_push():
    print("Commit + Push avec PAT...")
    
    # FIX IDENTITY CORRECT
    subprocess.run(["git", "config", "user.email", "github-actions@github.com"])
    subprocess.run(["git", "config", "user.name", "GitHub Actions Bot"])
    
    # Add le dossier data/
    subprocess.run(["git", "add", DATA_FOLDER])
    
    # Commit
    commit = subprocess.run(["git", "commit", "-m", f"Update prix {datetime.now().strftime('%Y-%m-%d')}"], 
                          capture_output=True, text=True)
    
    if commit.returncode != 0 or "nothing to commit" in commit.stdout:
        print("Rien à commiter (fichiers identiques)")
        return
    
    # Push avec PAT
    repo = os.getenv('GITHUB_REPOSITORY')
    url = f"https://{PUSH_TOKEN}@github.com/{repo}.git"
    push = subprocess.run(["git", "push", url, "HEAD:main"], capture_output=True, text=True)
    
    if push.returncode == 0:
        print("PUSH RÉUSSI ! Nouveau commit sur main")
    else:
        print("ERREUR PUSH:", push.stderr)

# ====================== MAIN ======================
if __name__ == "__main__":
    print("DÉBUT PIPELINE -", datetime.now().strftime("%Y-%m-%d %H:%M"))
    collect_yfinance()
    collect_tiingo()
    commit_and_push()
    print("TERMINÉ – PRIX DANS GIT REPO")
