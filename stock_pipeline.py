# -*- coding: utf-8 -*-
"""
PIPELINE COMPLÈTE : yFinance → Nettoyage → Features → Analyse → Git
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import subprocess
import json

# ====================== CONFIG ======================
PUSH_TOKEN = os.getenv('PUSH_TOKEN')

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Symboles à tracker
SYMBOLS = ["AAPL", "TSLA", "MSFT", "BTC-USD", "GOOGL", "NVDA", "AMZN", "META"]

# ====================== NETTOYAGE DES DONNÉES ======================
def clean_data(df, symbol):
    """Nettoyage des données financières"""
    df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    
    # Gestion des valeurs manquantes
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Supprimer les données invalides
    df = df[df['Close'] > 0]
    df = df.dropna(subset=['Close'])
    
    # Format standard
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['symbol'] = symbol
    
    return df

# ====================== FEATURES TECHNIQUES ======================
def create_technical_features(df, symbol):
    """Création des features techniques pour le ML"""
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. RETOURS ET MOUVEMENTS
    df['daily_return'] = df['Close'].pct_change()
    df['price_change'] = df['Close'].diff()
    
    # 2. VOLATILITÉ
    df['volatility_5'] = df['daily_return'].rolling(5).std()
    df['volatility_10'] = df['daily_return'].rolling(10).std()
    df['volatility_20'] = df['daily_return'].rolling(20).std()
    
    # 3. MOYENNES MOBILES
    for window in [5, 10, 20, 50]:
        df[f'MA_{window}'] = df['Close'].rolling(window).mean()
        df[f'price_vs_MA_{window}'] = (df['Close'] / df[f'MA_{window}']) - 1
    
    # 4. RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI_14'] = calculate_rsi(df['Close'])
    
    # 5. VOLUME ANALYSIS
    df['volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_MA_5']
    
    # 6. PRICE RANGE AND MOMENTUM
    df['price_range'] = (df['High'] - df['Low']) / df['Close']
    df['momentum_5'] = df['Close'].pct_change(5)
    df['momentum_10'] = df['Close'].pct_change(10)
    
    # 7. TARGET VARIABLES (pour la prédiction)
    df['target_3_days'] = (df['Close'].shift(-3) / df['Close']) - 1
    df['target_7_days'] = (df['Close'].shift(-7) / df['Close']) - 1
    df['target_direction_3d'] = (df['target_3_days'] > 0).astype(int)
    df['target_direction_7d'] = (df['target_7_days'] > 0).astype(int)
    
    # 8. TREND INDICATORS
    df['trend_5'] = (df['Close'] > df['MA_5']).astype(int)
    df['trend_20'] = (df['Close'] > df['MA_20']).astype(int)
    
    # 9. SUPPORT/RESISTANCE
    df['resistance_20'] = df['High'].rolling(20).max()
    df['support_20'] = df['Low'].rolling(20).min()
    df['distance_to_resistance'] = (df['resistance_20'] - df['Close']) / df['Close']
    df['distance_to_support'] = (df['Close'] - df['support_20']) / df['Close']
    
    # Nettoyer les NaN créés par les rolling windows
    df = df.dropna()
    
    print(f"✅ {symbol}: {len(df)} lignes, {len(df.columns)} features")
    return df

# ====================== ANALYSE ET RAPPORTS ======================
def generate_analysis_report(all_data):
    """Génère un rapport d'analyse des données"""
    print("📊 Génération du rapport d'analyse...")
    
    analysis_report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks_analyzed": [],
        "summary": {}
    }
    
    for symbol in all_data['symbol'].unique():
        stock_data = all_data[all_data['symbol'] == symbol]
        
        stock_analysis = {
            "symbol": symbol,
            "data_points": len(stock_data),
            "date_range": {
                "start": stock_data['date'].min(),
                "end": stock_data['date'].max()
            },
            "price_analysis": {
                "current_price": stock_data['Close'].iloc[-1],
                "price_change_7d": stock_data['Close'].pct_change(7).iloc[-1] * 100,
                "price_change_30d": stock_data['Close'].pct_change(30).iloc[-1] * 100,
                "volatility": stock_data['volatility_20'].iloc[-1] * 100
            },
            "technical_indicators": {
                "rsi": stock_data['RSI_14'].iloc[-1],
                "trend": "BULLISH" if stock_data['trend_20'].iloc[-1] == 1 else "BEARISH",
                "volume_trend": "HIGH" if stock_data['volume_ratio'].iloc[-1] > 1.2 else "NORMAL"
            }
        }
        
        analysis_report["stocks_analyzed"].append(stock_analysis)
    
    # Sauvegarder le rapport
    with open(f"{DATA_FOLDER}/analysis_report.json", "w") as f:
        json.dump(analysis_report, f, indent=2)
    
    # Sauvegarder un résumé CSV
    summary_df = pd.DataFrame([{
        'symbol': stock['symbol'],
        'current_price': stock['price_analysis']['current_price'],
        '7d_change_%': stock['price_analysis']['price_change_7d'],
        '30d_change_%': stock['price_analysis']['price_change_30d'],
        'RSI': stock['technical_indicators']['rsi'],
        'trend': stock['technical_indicators']['trend']
    } for stock in analysis_report["stocks_analyzed"]])
    
    summary_df.to_csv(f"{DATA_FOLDER}/market_summary.csv", index=False)
    print("✅ Rapport d'analyse généré")

# ====================== DÉTECTION D'ANOMALIES ======================
def detect_anomalies(all_data):
    """Détecte les anomalies dans les données"""
    print("🔍 Détection des anomalies...")
    
    anomalies = []
    
    for symbol in all_data['symbol'].unique():
        stock_data = all_data[all_data['symbol'] == symbol]
        
        # Détection des volumes anormaux
        volume_threshold = stock_data['Volume'].quantile(0.95)
        high_volume_days = stock_data[stock_data['Volume'] > volume_threshold]
        
        # Détection des mouvements de prix extrêmes
        price_move_threshold = stock_data['daily_return'].abs().quantile(0.95)
        extreme_moves = stock_data[stock_data['daily_return'].abs() > price_move_threshold]
        
        if not high_volume_days.empty:
            for _, row in high_volume_days.iterrows():
                anomalies.append({
                    'symbol': symbol,
                    'date': row['date'],
                    'type': 'HIGH_VOLUME',
                    'value': row['Volume'],
                    'message': f"Volume anormalement élevé: {row['Volume']:,.0f}"
                })
        
        if not extreme_moves.empty:
            for _, row in extreme_moves.iterrows():
                anomalies.append({
                    'symbol': symbol,
                    'date': row['date'],
                    'type': 'EXTREME_MOVE',
                    'value': row['daily_return'] * 100,
                    'message': f"Mouvement de prix extrême: {row['daily_return']*100:.2f}%"
                })
    
    # Sauvegarder les anomalies
    if anomalies:
        anomalies_df = pd.DataFrame(anomalies)
        anomalies_df.to_csv(f"{DATA_FOLDER}/detected_anomalies.csv", index=False)
        print(f"✅ {len(anomalies)} anomalies détectées")
    else:
        print("✅ Aucune anomalie détectée")

# ====================== COLLECTE YFINANCE ======================
def collect_yfinance():
    """Collecte principale des données yFinance"""
    all_data = []
    
    print("📈 Collecte des données yFinance...")
    
    for symbol in SYMBOLS:
        try:
            # Collecte des données (2 ans pour avoir assez d'historique)
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y").reset_index()
            
            # Préparation des données
            df['symbol'] = symbol
            df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
            
            # Nettoyage et features
            df_clean = clean_data(df, symbol)
            df_features = create_technical_features(df_clean, symbol)
            
            # Sauvegarde individuelle
            df_features.to_csv(f"{DATA_FOLDER}/{symbol}_with_features.csv", index=False)
            all_data.append(df_features)
            
            print(f"✅ {symbol} - {len(df_features)} jours de données")
            
        except Exception as e:
            print(f"❌ Erreur avec {symbol}: {e}")
    
    if all_data:
        # Sauvegarde du fichier combiné
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.to_csv(f"{DATA_FOLDER}/ALL_STOCKS_with_features.csv", index=False)
        
        # Actions supplémentaires
        generate_analysis_report(combined_data)
        detect_anomalies(combined_data)
        
        print(f"✅ Collecte terminée - {len(combined_data)} lignes au total")
        return combined_data
    else:
        raise Exception("Aucune donnée collectée")

# ====================== GIT ACTIONS ======================
def git_actions():
    """Gère les actions Git"""
    print("🔄 Actions Git...")
    
    # Configuration Git
    subprocess.run(["git", "config", "user.email", "github-actions@github.com"])
    subprocess.run(["git", "config", "user.name", "GitHub Actions Bot"])
    
    # Ajout de tous les fichiers data
    subprocess.run(["git", "add", DATA_FOLDER])
    
    # Commit
    commit_msg = f"🤖 Update data & analysis {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    commit = subprocess.run(["git", "commit", "-m", commit_msg], 
                          capture_output=True, text=True)
    
    if commit.returncode != 0 or "nothing to commit" in commit.stdout:
        print("✅ Aucun changement à commiter")
        return False
    
    # Push
    repo = os.getenv('GITHUB_REPOSITORY')
    url = f"https://{PUSH_TOKEN}@github.com/{repo}.git"
    push = subprocess.run(["git", "push", url, "HEAD:main"], capture_output=True, text=True)
    
    if push.returncode == 0:
        print("✅ Push réussi !")
        return True
    else:
        print("❌ Erreur push:", push.stderr)
        return False

# ====================== MAIN ======================
if __name__ == "__main__":
    print("🚀 DÉBUT PIPELINE AVANCÉE")
    print("=" * 50)
    
    try:
        # Collecte et traitement des données
        data = collect_yfinance()
        
        print("\n" + "=" * 50)
        print("📦 Données prêtes pour le ML:")
        print(f"   • Fichiers individuels: {[f'{s}_with_features.csv' for s in SYMBOLS]}")
        print(f"   • Fichier combiné: ALL_STOCKS_with_features.csv")
        print(f"   • Rapport: analysis_report.json")
        print(f"   • Résumé: market_summary.csv")
        print(f"   • Anomalies: detected_anomalies.csv")
        
        # Actions Git
        print("\n" + "=" * 50)
        git_success = git_actions()
        
        if git_success:
            print("\n🎉 PIPELINE TERMINÉE AVEC SUCCÈS !")
        else:
            print("\nℹ️  Pipeline terminée (aucun changement)")
            
    except Exception as e:
        print(f"\n💥 ERREUR CRITIQUE: {e}")
        raise
