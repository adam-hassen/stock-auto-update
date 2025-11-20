# -*- coding: utf-8 -*-
"""
PIPELINE COMPLÈTE : yFinance → Nettoyage → Features → 3 versions → Git
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
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

# ====================== FEATURES ESSENTIELLES ======================
def create_essential_features(df, symbol):
    """Features les plus importantes seulement"""
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. RETOURS
    df['daily_return'] = df['Close'].pct_change()
    
    # 2. VOLATILITÉ
    df['volatility_10'] = df['daily_return'].rolling(10).std()
    
    # 3. MOYENNES MOBILES (seulement 3)
    for window in [5, 20, 50]:
        df[f'MA_{window}'] = df['Close'].rolling(window).mean()
    
    # 4. RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI_14'] = calculate_rsi(df['Close'])
    
    # 5. VOLUME
    df['volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_MA_5']
    
    # 6. TARGETS SIMPLES
    df['target_3_days'] = (df['Close'].shift(-3) / df['Close']) - 1
    df['target_direction'] = (df['target_3_days'] > 0).astype(int)
    
    # Nettoyer les NaN
    df = df.dropna()
    
    print(f"✅ {symbol}: {len(df)} lignes avec features")
    return df

# ====================== SAUVEGARDE DES 3 VERSIONS ======================
def save_all_versions(df_clean, df_features, symbol, all_cleaned_data, all_features_data):
    """Sauvegarde les 3 versions des données"""
    
    # 1. Fichier individuel avec features
    df_features.to_csv(f"{DATA_FOLDER}/{symbol}.csv", index=False)
    
    # Ajouter aux données combinées
    all_cleaned_data.append(df_clean)
    all_features_data.append(df_features)
    
    print(f"💾 {symbol}.csv sauvegardé")

# ====================== SAUVEGARDE DES FICHIERS COMBINÉS ======================
def save_combined_files(all_cleaned_data, all_features_data):
    """Sauvegarde les fichiers combinés"""
    
    # 2. Fichier combiné NETTOYÉ seulement
    combined_cleaned = pd.concat(all_cleaned_data, ignore_index=True)
    combined_cleaned.to_csv(f"{DATA_FOLDER}/ALL_CLEANED.csv", index=False)
    print("💾 ALL_CLEANED.csv sauvegardé")
    
    # 3. Fichier combiné avec FEATURES
    combined_features = pd.concat(all_features_data, ignore_index=True)
    combined_features.to_csv(f"{DATA_FOLDER}/ALL_FEATURES.csv", index=False)
    print("💾 ALL_FEATURES.csv sauvegardé")
    
    return combined_cleaned, combined_features

# ====================== RAPPORT SIMPLIFIÉ ======================
def generate_simple_report(combined_cleaned, combined_features):
    """Rapport simple des données disponibles"""
    print("📊 Génération du rapport...")
    
    report_data = []
    
    for symbol in SYMBOLS:
        symbol_data = combined_cleaned[combined_cleaned['symbol'] == symbol]
        if len(symbol_data) > 0:
            report_data.append({
                'symbol': symbol,
                'start_date': symbol_data['date'].min(),
                'end_date': symbol_data['date'].max(),
                'days_count': len(symbol_data),
                'last_price': symbol_data['Close'].iloc[-1],
                'last_volume': symbol_data['Volume'].iloc[-1]
            })
    
    # Sauvegarder le rapport
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(f"{DATA_FOLDER}/data_report.csv", index=False)
        print("✅ Rapport généré")

# ====================== COLLECTE YFINANCE ======================
def collect_yfinance():
    """Collecte principale des données yFinance"""
    all_cleaned_data = []
    all_features_data = []
    
    print("📈 Collecte des données yFinance...")
    
    for symbol in SYMBOLS:
        try:
            # Collecte des données
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y").reset_index()
            
            # Préparation des données
            df['symbol'] = symbol
            df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
            
            # Nettoyage
            df_clean = clean_data(df, symbol)
            
            # Features
            df_features = create_essential_features(df_clean.copy(), symbol)
            
            # Sauvegarde des 3 versions
            save_all_versions(df_clean, df_features, symbol, all_cleaned_data, all_features_data)
            
        except Exception as e:
            print(f"❌ Erreur avec {symbol}: {e}")
    
    # Sauvegarde des fichiers combinés
    combined_cleaned, combined_features = save_combined_files(all_cleaned_data, all_features_data)
    
    # Générer le rapport
    generate_simple_report(combined_cleaned, combined_features)
    
    print(f"✅ Collecte terminée")
    print(f"   • Fichiers individuels: {len(SYMBOLS)} actions")
    print(f"   • Fichier combiné nettoyé: {len(combined_cleaned)} lignes")
    print(f"   • Fichier combiné avec features: {len(combined_features)} lignes")

# ====================== GIT ACTIONS ======================
def git_actions():
    """Gère les actions Git"""
    print("🔄 Actions Git...")
    
    subprocess.run(["git", "config", "user.email", "github-actions@github.com"])
    subprocess.run(["git", "config", "user.name", "GitHub Actions Bot"])
    
    subprocess.run(["git", "add", DATA_FOLDER])
    
    commit_msg = f"Update données {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    commit = subprocess.run(["git", "commit", "-m", commit_msg], 
                          capture_output=True, text=True)
    
    if commit.returncode != 0 or "nothing to commit" in commit.stdout:
        print("✅ Aucun changement à commiter")
        return False
    
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
    print("🚀 DÉBUT PIPELINE COMPLÈTE")
    print("=" * 50)
    
    try:
        # Collecte et traitement des données
        collect_yfinance()
        
        print("\n" + "=" * 50)
        print("📦 FICHIERS CRÉÉS:")
        print("   1. FICHIERS INDIVIDUELS (nettoyés + features):")
        for symbol in SYMBOLS:
            file_path = f"{DATA_FOLDER}/{symbol}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"      • {symbol}.csv - {len(df)} lignes")
        
        print("   2. FICHIERS COMBINÉS:")
        print(f"      • ALL_CLEANED.csv - données nettoyées seulement")
        print(f"      • ALL_FEATURES.csv - données avec features")
        print(f"   3. RAPPORT:")
        print(f"      • data_report.csv")
        
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
