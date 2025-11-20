# -*- coding: utf-8 -*-
"""
PIPELINE PRIX → NETTOYAGE → FEATURES → GIT REPO
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from tiingo import TiingoClient
import subprocess

# ====================== CONFIG ======================
TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY manquante !")

PUSH_TOKEN = os.getenv('PUSH_TOKEN')
if not PUSH_TOKEN:
    raise ValueError("PUSH_TOKEN manquant !")

DATA_FOLDER = "data"
CLEANED_FOLDER = "data/cleaned"
FEATURES_FOLDER = "data/features"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)
os.makedirs(FEATURES_FOLDER, exist_ok=True)

# ====================== NETTOYAGE DES DONNÉES ======================
def clean_data(df, symbol):
    """
    Nettoyage complet des données financières
    """
    # Supprimer les doublons
    df = df.drop_duplicates(subset=['date'], keep='last')
    
    # Trier par date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Vérifier les valeurs manquantes
    print(f"Valeurs manquantes pour {symbol}:")
    print(df.isnull().sum())
    
    # Remplir les valeurs manquantes (méthode forward fill pour les données financières)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Supprimer les lignes où le prix de clôture est manquant
    df = df.dropna(subset=['Close'])
    
    # Vérifier les valeurs aberrantes (prix négatifs)
    df = df[df['Close'] > 0]
    df = df[df['Volume'] >= 0]
    
    # Normaliser les formats de date
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Ajouter la colonne symbol si elle n'existe pas
    if 'symbol' not in df.columns:
        df['symbol'] = symbol
    
    print(f"Données nettoyées pour {symbol}: {len(df)} lignes")
    return df

# ====================== CRÉATION DES FEATURES ======================
def create_features(df, symbol):
    """
    Création des features techniques pour le modèle ML
    """
    # S'assurer que les données sont triées par date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Features de base
    df['price_change'] = df['Close'].pct_change()
    df['daily_return'] = df['Close'].pct_change() * 100
    df['high_low_ratio'] = df['High'] / df['Low']
    df['volume_change'] = df['Volume'].pct_change()
    
    # Moyennes mobiles
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Ratios des moyennes mobiles
    df['MA_5_ratio'] = df['Close'] / df['MA_5']
    df['MA_10_ratio'] = df['Close'] / df['MA_10']
    df['MA_20_ratio'] = df['Close'] / df['MA_20']
    
    # RSI (Relative Strength Index)
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI_14'] = calculate_rsi(df['Close'])
    
    # MACD (Moving Average Convergence Divergence)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Volatilité
    df['volatility_5'] = df['Close'].pct_change().rolling(window=5).std()
    df['volatility_10'] = df['Close'].pct_change().rolling(window=10).std()
    df['volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
    
    # Features de momentum
    df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    
    # Target variable (prix futur - pour la prédiction)
    df['target_5_days'] = df['Close'].shift(-5) / df['Close'] - 1  # Rendement dans 5 jours
    df['target_direction'] = (df['target_5_days'] > 0).astype(int)  # 1 si hausse, 0 si baisse
    
    # Features temporelles
    df['date_dt'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date_dt'].dt.dayofweek
    df['month'] = df['date_dt'].dt.month
    df['quarter'] = df['date_dt'].dt.quarter
    
    # Supprimer les NaN créés par les rolling windows
    df = df.dropna()
    
    print(f"Features créées pour {symbol}: {len(df)} lignes, {len(df.columns)} colonnes")
    return df

# ====================== COLLECTE YFINANCE ======================
def collect_yfinance():
    symbols = ["AAPL", "TSLA", "MSFT", "BTC-USD", "GOOGL"]
    all_raw_data = []
    all_cleaned_data = []
    all_features_data = []
    
    print("Collecte yfinance...")
    for s in symbols:
        # Collecte
        df = yf.Ticker(s).history(period="2y").reset_index()  # 2 ans pour avoir assez de données après nettoyage
        df['symbol'] = s
        df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
        all_raw_data.append(df)
        
        # Sauvegarde raw
        df.to_csv(f"{DATA_FOLDER}/{s}.csv", index=False)
        
        # Nettoyage
        df_cleaned = clean_data(df, s)
        df_cleaned.to_csv(f"{CLEANED_FOLDER}/{s}_cleaned.csv", index=False)
        all_cleaned_data.append(df_cleaned)
        
        # Features
        df_features = create_features(df_cleaned, s)
        df_features.to_csv(f"{FEATURES_FOLDER}/{s}_features.csv", index=False)
        all_features_data.append(df_features)
    
    # Sauvegarde des données combinées
    pd.concat(all_raw_data).to_csv(f"{DATA_FOLDER}/ALL_YFINANCE.csv", index=False)
    pd.concat(all_cleaned_data).to_csv(f"{CLEANED_FOLDER}/ALL_YFINANCE_cleaned.csv", index=False)
    pd.concat(all_features_data).to_csv(f"{FEATURES_FOLDER}/ALL_YFINANCE_features.csv", index=False)
    print("yfinance OK - Données nettoyées et features créées")

# ====================== COLLECTE TIINGO ======================
def collect_tiingo():
    client = TiingoClient({'api_key': TIINGO_API_KEY, 'session': True})
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL"]
    all_raw_data = []
    all_cleaned_data = []
    all_features_data = []
    
    print("Collecte Tiingo...")
    for s in symbols:
        try:
            # Collecte
            df = client.get_dataframe(s, frequency='daily', 
                                    startDate=datetime.now().replace(year=datetime.now().year-2))
            df = df.reset_index()
            df['symbol'] = s
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
            df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']
            all_raw_data.append(df)
            
            # Sauvegarde raw
            df.to_csv(f"{DATA_FOLDER}/tiingo/{s}_prices.csv", index=False)
            
            # Nettoyage
            df_cleaned = clean_data(df, s)
            df_cleaned.to_csv(f"{CLEANED_FOLDER}/tiingo/{s}_cleaned.csv", index=False)
            all_cleaned_data.append(df_cleaned)
            
            # Features
            df_features = create_features(df_cleaned, s)
            df_features.to_csv(f"{FEATURES_FOLDER}/tiingo/{s}_features.csv", index=False)
            all_features_data.append(df_features)
            
        except Exception as e:
            print(f"Erreur avec {s}: {e}")
    
    # Sauvegarde des données combinées
    if all_raw_data:
        pd.concat(all_raw_data).to_csv(f"{DATA_FOLDER}/tiingo/ALL_TIINGO.csv", index=False)
    if all_cleaned_data:
        pd.concat(all_cleaned_data).to_csv(f"{CLEANED_FOLDER}/tiingo/ALL_TIINGO_cleaned.csv", index=False)
    if all_features_data:
        pd.concat(all_features_data).to_csv(f"{FEATURES_FOLDER}/tiingo/ALL_TIINGO_features.csv", index=False)
    print("Tiingo OK - Données nettoyées et features créées")

# ====================== GIT COMMIT + PUSH ======================
def commit_and_push():
    print("Commit + Push avec PAT...")
    
    # Configuration Git
    subprocess.run(["git", "config", "user.email", "github-actions@github.com"])
    subprocess.run(["git", "config", "user.name", "GitHub Actions Bot"])
    
    # Add tous les dossiers de données
    subprocess.run(["git", "add", DATA_FOLDER, CLEANED_FOLDER, FEATURES_FOLDER])
    
    # Commit
    commit = subprocess.run(["git", "commit", "-m", f"Update données {datetime.now().strftime('%Y-%m-%d')} - avec nettoyage et features"], 
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
    print("DÉBUT PIPELINE COMPLÈTE -", datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    # Créer les sous-dossiers
    os.makedirs(f"{CLEANED_FOLDER}/tiingo", exist_ok=True)
    os.makedirs(f"{FEATURES_FOLDER}/tiingo", exist_ok=True)
    
    collect_yfinance()
    collect_tiingo()
    commit_and_push()
    
    print("TERMINÉ – DONNÉES NETTOYÉES ET FEATURES CRÉÉES")
