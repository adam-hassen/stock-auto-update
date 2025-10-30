# -*- coding: utf-8 -*-
"""
PIPELINE PRIX SEULEMENT
yfinance + Tiingo → Google Drive
Auto tous les jours
"""

import os
import pandas as pd
from datetime import datetime
import yfinance as yf
from tiingo import TiingoClient
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ====================== CONFIG ======================
SCOPES = ['https://www.googleapis.com/auth/drive.file']
TOKEN_FILE = 'token.json'
FOLDER_ID = '1LOk5epELmfSV0U_XnKPN_DWG5Vql3xGa'

TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY manquante !")

# ====================== GOOGLE DRIVE ======================
def authenticate_google_drive():
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build('drive', 'v3', credentials=creds)

# ====================== YFINANCE ======================
def collect_yfinance():
    symbols = ["AAPL", "TSLA", "MSFT", "BTC-USD", "GOOGL"]
    path = "/tmp/stock_data"
    os.makedirs(path, exist_ok=True)
    all_data = []
    print("yfinance → en cours...")
    for s in symbols:
        print(f"  → {s}")
        df = yf.Ticker(s).history(period="1y").reset_index()
        df['symbol'] = s
        df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
        all_data.append(df)
        df.to_csv(f"{path}/{s}.csv", index=False)
    pd.concat(all_data).to_csv(f"{path}/ALL_YFINANCE.csv", index=False)
    print(f"yfinance : {len(all_data)} fichiers OK")
    return path

# ====================== TIINGO ======================
def collect_tiingo():
    client = TiingoClient({'api_key': TIINGO_API_KEY, 'session': True})
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL"]
    path = "/tmp/stock_data/tiingo"
    os.makedirs(path, exist_ok=True)
    all_data = []
    print("Tiingo → en cours...")
    for s in symbols:
        print(f"  → {s}")
        df = client.get_dataframe(s, frequency='daily', startDate=datetime.now().replace(year=datetime.now().year-1))
        df = df.reset_index()
        df['symbol'] = s
        df['source'] = 'tiingo'
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'source']]
        df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol', 'source']
        all_data.append(df)
        df.to_csv(f"{path}/{s}_prices.csv", index=False)
    pd.concat(all_data).to_csv(f"{path}/ALL_TIINGO.csv", index=False)
    print(f"Tiingo : {len(all_data)} fichiers OK")
    return "/tmp/stock_data"

# ====================== UPLOAD ======================
def upload_to_drive(local_path):
    print("Upload sur Google Drive...")
    service = authenticate_google_drive()
    for root, _, files in os.walk(local_path):
        for file in files:
            file_path = os.path.join(root, file)
            name = os.path.relpath(file_path, local_path).replace(os.sep, '_')
            try:
                service.files().create(
                    body={'name': name, 'parents': [FOLDER_ID]},
                    media_body=MediaFileUpload(file_path)
                ).execute()
                print(f"   Uploaded: {name}")
            except Exception as e:
                print(f"   Erreur {file}: {e}")

# ====================== MAIN ======================
if __name__ == "__main__":
    print("DÉBUT PIPELINE PRIX -", datetime.now().strftime("%Y-%m-%d %H:%M"))
    collect_yfinance()
    collect_tiingo()
    upload_to_drive("/tmp/stock_data")
    print("PIPELINE TERMINÉ ! TOUT EST DANS DRIVE.")
