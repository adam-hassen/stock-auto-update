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
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive.file']
TOKEN_FILE = 'token.json'
FOLDER_ID = '1LOk5epELmfSV0U_XnKPN_DWG5Vql3xGa'

TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY manquante !")

def authenticate_google_drive():
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build('drive', 'v3', credentials=creds)

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

def upload_to_drive(local_path):
    print("Upload Drive...")
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
                print(f"  OK: {name}")
            except Exception as e:
                print(f"  ERREUR: {e}")

if __name__ == "__main__":
    print("DÉBUT -", datetime.now().strftime("%H:%M"))
    collect_yfinance()
    collect_tiingo()
    upload_to_drive("/tmp/stock_data")
    print("TERMINÉ – PRIX DANS DRIVE")
