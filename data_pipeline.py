import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ====================== CONFIGURATION ======================
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

SYMBOLS = ["AAPL", "TSLA", "MSFT", "BTC-USD", "GOOGL", "NVDA", "AMZN", "META"]
YEARS_BACK = 2   # 2 ANS DE DONNEES

# ====================== FEATURES ======================
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features(df):
    df['daily_return'] = df['Close'].pct_change()
    df['volatility_10d'] = df['daily_return'].rolling(10).std()

    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()

    df['RSI_14'] = calculate_rsi(df['Close'])

    df['volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_MA_5']

    df['target_3d'] = (df['Close'].shift(-3) / df['Close']) - 1

    return df.dropna().reset_index(drop=True)

# ====================== COLLECTE ======================
def collect_data():
    all_actions = []

    print("COLLECTE DES DONNEES FINANCIERES")
    print("=" * 50)

    for symbol in SYMBOLS:
        try:
            print(f"Traitement de {symbol}")

            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * YEARS_BACK)

            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                print(f"  Aucune donnee pour {symbol}")
                continue

            df = df.reset_index()
            df['date'] = df['Date'].dt.strftime('%Y-%m-%d')

            df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.dropna()
            df = df[df['Close'] > 0]
            df = df.sort_values('date').reset_index(drop=True)

            df = add_features(df)
            df['symbol'] = symbol

            # FICHIER INDIVIDUEL
            df.to_csv(f"{DATA_FOLDER}/{symbol}.csv", index=False)
            print(f"  {symbol}.csv sauvegarde ({len(df)} lignes)")

            all_actions.append(df)

        except Exception as e:
            print(f"  Erreur {symbol}: {e}")

    # FICHIER GLOBAL
    if all_actions:
        df_all = pd.concat(all_actions, ignore_index=True)
        df_all.to_csv(f"{DATA_FOLDER}/ALL_ACTIONS.csv", index=False)
        print("\nFICHIER GLOBAL CREE: ALL_ACTIONS.csv")
        print(f"Total lignes: {len(df_all)}")

# ====================== EXECUTION ======================
if __name__ == "__main__":
    print("LANCEMENT PIPELINE FINANCIER")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    collect_data()
    print("PIPELINE TERMINE AVEC SUCCES")
