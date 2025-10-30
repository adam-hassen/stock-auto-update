# -*- coding: utf-8 -*-
"""
stock_pipeline.py
Pipeline automatique : yfinance + Tiingo → Google Drive
Exécution quotidienne via GitHub Actions
"""

import os
import pandas as pd
from datetime import datetime
import yfinance as yf
from tiingo import TiingoClient
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ====================== CONFIGURATION ======================
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account.json'
FOLDER_ID = '1LOk5epELmfSV0U_XnKPN_DWG5Vql3xGa'

# Récupère la clé Tiingo depuis secrets
TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY non trouvée !")

# ====================== GOOGLE DRIVE AUTH (SERVICE ACCOUNT) ======================
def authenticate_google_drive():
    service_json = os.getenv('SERVICE_ACCOUNT_JSON')
    if not service_json:
        raise ValueError("SERVICE_ACCOUNT_JSON non trouvée !")
    with open(SERVICE_ACCOUNT_FILE, 'w') as f:
        f.write(service_json)
    creds = ServiceAccountCredentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

# ====================== COLLECTE YFINANCE ======================
def collect_yfinance():
    symbols = ["AAPL", "TSLA", "MSFT", "BTC-USD", "GOOGL"]
    save_path = "/tmp/stock_data"
    os.makedirs(save_path, exist_ok=True)

    all_data = []
    print("Collecte yfinance en cours...")

    for symbol in symbols:
        print(f"  → {symbol}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y")
        df = df.reset_index()
        df['symbol'] = symbol
        df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
        all_data.append(df)
        df.to_csv(f"{save_path}/{symbol}.csv", index=False)

    full_df = pd.concat(all_data)
    full_df.to_csv(f"{save_path}/ALL_STOCKS.csv", index=False)
    print(f"yfinance : {len(full_df)} lignes sauvegardées.")
    return save_path

# ====================== COLLECTE TIINGO ======================
def collect_tiingo():
    config = {'api_key': TIINGO_API_KEY, 'session': True}
    client = TiingoClient(config)
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL"]
    tiingo_path = "/tmp/stock_data/tiingo_data"
    os.makedirs(tiingo_path, exist_ok=True)

    print("Collecte Tiingo en cours...")

    all_prices = []

    for symbol in symbols:
        print(f"  → {symbol}")

        # Prix
        try:
            end_date = datetime.now()
            start_date = end_date.replace(year=end_date.year - 1)
            df_prices = client.get_dataframe(symbol, startDate=start_date, endDate=end_date, frequency='daily')
            df_prices = df_prices.reset_index()
            df_prices['symbol'] = symbol
            df_prices['source'] = 'tiingo'
            df_prices = df_prices[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'source']]
            df_prices.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol', 'source']
            df_prices.to_csv(f"{tiingo_path}/{symbol}_prices.csv", index=False)
            all_prices.append(df_prices)
            print(f"     Prix : {len(df_prices)} lignes")
        except Exception as e:
            print(f"     Erreur prix {symbol}: {e}")

        # News (FIX : get_ticker_news)
        try:
            news = client.get_ticker_news(symbol, limit=5)
            news_df = pd.DataFrame(news)
            news_df['symbol'] = symbol
            news_df['source'] = 'tiingo'
            news_df.to_csv(f"{tiingo_path}/{symbol}_news.csv", index=False)
            print(f"     News : {len(news_df)} articles")
        except Exception as e:
            print(f"     Erreur news {symbol}: {e}")

    # Fusion prix
    if all_prices:
        full_tiingo = pd.concat(all_prices)
        full_tiingo.to_csv(f"{tiingo_path}/ALL_TIINGO_PRICES.csv", index=False)

    return "/tmp/stock_data"

# ====================== UPLOAD VERS GOOGLE DRIVE ======================
def upload_to_drive(local_path):
    print("Connexion à Google Drive...")
    service = authenticate_google_drive()

    print("Upload des fichiers...")
    for root, _, files in os.walk(local_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, local_path)
            file_metadata = {
                'name': relative_path.replace(os.sep, '_'),
                'parents': [FOLDER_ID]
            }
            media = MediaFileUpload(file_path)
            try:
                service.files().create(body=file_metadata, media_body=media).execute()
                print(f"   Uploaded: {relative_path}")
            except Exception as e:
                print(f"   Erreur upload {file}: {e}")

# ====================== MAIN ======================
if __name__ == "__main__":
    print("DÉBUT DU PIPELINE AUTOMATIQUE -", datetime.now().strftime("%Y-%m-%d %H:%M"))

    # 1. yfinance
    collect_yfinance()

    # 2. Tiingo
    collect_tiingo()

    # 3. Upload
    upload_to_drive("/tmp/stock_data")

    print("PIPELINE TERMINÉ !")
