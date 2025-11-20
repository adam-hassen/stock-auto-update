# -*- coding: utf-8 -*-
"""
PIPELINE COMPLÈTE : yFinance → Nettoyage → Features → Anomalies → 3 versions → Git
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import subprocess
import json
import warnings
warnings.filterwarnings('ignore')

# ====================== CONFIG ======================
PUSH_TOKEN = os.getenv('PUSH_TOKEN')
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Symboles à tracker
SYMBOLS = ["AAPL", "TSLA", "MSFT", "BTC-USD", "GOOGL", "NVDA", "AMZN", "META"]

# ====================== FONCTION POUR CONVERTIR LES TYPES NUMPY ======================
def convert_numpy_types(obj):
    """Convertit les types NumPy en types Python natifs pour JSON"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# ====================== DÉTECTION D'ANOMALIES AVANCÉE ======================
def calculate_zscore(data):
    """Calcule le Z-score sans scipy"""
    if len(data) == 0:
        return np.array([])
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.zeros(len(data))
    return np.abs((data - mean) / std)

def detect_anomalies_avancee(df, symbol):
    """Détection avancée des anomalies dans les données financières"""
    
    anomalies = []
    
    # 1. ANOMALIES DE PRIX (Mouvements extrêmes)
    returns = df['Close'].pct_change().dropna()
    if len(returns) > 0:
        z_scores_returns = calculate_zscore(returns)
        extreme_returns_idx = np.where(z_scores_returns > 3)[0]
        
        for idx in extreme_returns_idx:
            if idx < len(df):
                date_anomalie = df.iloc[idx]['date']
                prix = float(df.iloc[idx]['Close'])
                rendement = float(returns.iloc[idx] * 100)
                
                anomalies.append({
                    'symbol': symbol,
                    'date': date_anomalie,
                    'type': 'MOUVEMENT_EXTREME',
                    'severite': 'HAUTE',
                    'valeur': rendement,
                    'description': f"Mouvement de prix extrême: {rendement:.2f}%",
                    'prix_ce_jour': prix
                })
    
    # 2. ANOMALIES DE VOLUME (Volume anormalement élevé)
    volume_data = df['Volume'].dropna()
    if len(volume_data) > 0:
        volume_z_scores = calculate_zscore(volume_data)
        high_volume_idx = np.where(volume_z_scores > 2.5)[0]
        
        for idx in high_volume_idx:
            if idx < len(df):
                date_anomalie = df.iloc[idx]['date']
                volume = int(df.iloc[idx]['Volume'])
                volume_moyen = float(df['Volume'].mean())
                
                anomalies.append({
                    'symbol': symbol,
                    'date': date_anomalie,
                    'type': 'VOLUME_ANORMAL',
                    'severite': 'MOYENNE',
                    'valeur': volume,
                    'description': f"Volume anormal: {volume:,.0f} vs moyenne {volume_moyen:,.0f}",
                    'ratio_volume': float(volume / volume_moyen)
                })
    
    # 3. GAPS DE PRIX (Ouverture très différente de clôture précédente)
    df['overnight_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    gap_data = df['overnight_gap'].dropna()
    if len(gap_data) > 0:
        gap_z_scores = calculate_zscore(gap_data)
        gap_anomalies_idx = np.where(gap_z_scores > 2.5)[0]
        
        for idx in gap_anomalies_idx:
            if idx < len(df):
                date_anomalie = df.iloc[idx]['date']
                gap_pourcentage = float(df.iloc[idx]['overnight_gap'] * 100)
                
                anomalies.append({
                    'symbol': symbol,
                    'date': date_anomalie,
                    'type': 'GAP_OUVERTURE',
                    'severite': 'MOYENNE',
                    'valeur': gap_pourcentage,
                    'description': f"Gap d'ouverture: {gap_pourcentage:.2f}%",
                    'ouverture': float(df.iloc[idx]['Open']),
                    'cloture_precedente': float(df.iloc[idx-1]['Close']) if idx > 0 else None
                })
    
    # 4. ANOMALIES DE VOLATILITÉ
    if len(returns) > 10:
        volatilite_rolling = returns.rolling(window=10).std().dropna()
        if len(volatilite_rolling) > 0:
            vol_z_scores = calculate_zscore(volatilite_rolling)
            high_vol_idx = np.where(vol_z_scores > 2.5)[0]
            
            for idx in high_vol_idx:
                if idx + 10 < len(df):
                    date_anomalie = df.iloc[idx + 10]['date']
                    volatilite = float(volatilite_rolling.iloc[idx] * 100)
                    
                    anomalies.append({
                        'symbol': symbol,
                        'date': date_anomalie,
                        'type': 'VOLATILITE_EXTREME',
                        'severite': 'HAUTE',
                        'valeur': volatilite,
                        'description': f"Volatilité extrême: {volatilite:.2f}%",
                        'periode': '10 jours'
                    })
    
    # 5. CASSURES DE TENDANCE (Prix qui casse une MM importante)
    for window in [20, 50]:
        ma_col = f'MA_{window}'
        if ma_col in df.columns:
            # Détection de cassure au-dessus
            cassure_hausse = (df['Close'] > df[ma_col]) & (df['Close'].shift(1) <= df[ma_col].shift(1))
            # Détection de cassure en-dessous
            cassure_baisse = (df['Close'] < df[ma_col]) & (df['Close'].shift(1) >= df[ma_col].shift(1))
            
            cassure_idx = np.where(cassure_hausse | cassure_baisse)[0]
            
            for idx in cassure_idx:
                if idx < len(df):
                    date_anomalie = df.iloc[idx]['date']
                    direction = "HAUSSE" if cassure_hausse.iloc[idx] else "BAISSE"
                    
                    anomalies.append({
                        'symbol': symbol,
                        'date': date_anomalie,
                        'type': 'CASSURE_TENDANCE',
                        'severite': 'MOYENNE',
                        'valeur': int(window),
                        'description': f"Cassure {direction} de la MM{window}",
                        'prix': float(df.iloc[idx]['Close']),
                        f'MA_{window}': float(df.iloc[idx][ma_col])
                    })
    
    # 6. RSI EXTRÊME
    if 'RSI_14' in df.columns:
        rsi_data = df['RSI_14'].dropna()
        if len(rsi_data) > 0:
            rsi_extreme_haut = df['RSI_14'] > 80
            rsi_extreme_bas = df['RSI_14'] < 20
            
            rsi_extreme_idx = np.where(rsi_extreme_haut | rsi_extreme_bas)[0]
            
            for idx in rsi_extreme_idx:
                if idx < len(df):
                    date_anomalie = df.iloc[idx]['date']
                    rsi_valeur = float(df.iloc[idx]['RSI_14'])
                    condition = "SURACHAT" if rsi_extreme_haut.iloc[idx] else "SURVENDE"
                    
                    anomalies.append({
                        'symbol': symbol,
                        'date': date_anomalie,
                        'type': 'RSI_EXTREME',
                        'severite': 'MOYENNE',
                        'valeur': rsi_valeur,
                        'description': f"RSI en {condition}: {rsi_valeur:.1f}",
                        'niveau': condition
                    })
    
    return anomalies

def sauvegarder_anomalies(anomalies_par_action):
    """Sauvegarde toutes les anomalies détectées"""
    if not any(anomalies_par_action.values()):
        print("✅ Aucune anomalie détectée")
        return
    
    # Combiner toutes les anomalies
    toutes_anomalies = []
    for symbol, anomalies in anomalies_par_action.items():
        toutes_anomalies.extend(anomalies)
    
    # Sauvegarder en CSV
    df_anomalies = pd.DataFrame(toutes_anomalies)
    df_anomalies.to_csv(f"{DATA_FOLDER}/ANOMALIES_DETECTEES.csv", index=False)
    
    # Préparer le résumé avec conversion des types
    resume_anomalies = {
        'date_analyse': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_anomalies': len(toutes_anomalies),
        'par_type': convert_numpy_types(df_anomalies['type'].value_counts().to_dict()),
        'par_severite': convert_numpy_types(df_anomalies['severite'].value_counts().to_dict()),
        'par_action': convert_numpy_types(df_anomalies['symbol'].value_counts().to_dict()),
        'anomalies_recentes': convert_numpy_types(sorted(toutes_anomalies, 
                                   key=lambda x: x['date'], 
                                   reverse=True)[:10])
    }
    
    # Sauvegarder le résumé en JSON
    with open(f"{DATA_FOLDER}/resume_anomalies.json", "w") as f:
        json.dump(resume_anomalies, f, indent=2, ensure_ascii=False)
    
    print(f"✅ {len(toutes_anomalies)} anomalies détectées et sauvegardées")
    if resume_anomalies['par_type']:
        print(f"   📊 Répartition: {resume_anomalies['par_type']}")

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
                'days_count': int(len(symbol_data)),
                'last_price': float(symbol_data['Close'].iloc[-1]),
                'last_volume': int(symbol_data['Volume'].iloc[-1])
            })
    
    # Sauvegarder le rapport
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(f"{DATA_FOLDER}/data_report.csv", index=False)
        print("✅ Rapport généré")

# ====================== COLLECTE YFINANCE CORRIGÉE ======================
def collect_yfinance():
    """Collecte principale des données yFinance - VERSION CORRIGÉE"""
    all_cleaned_data = []
    all_features_data = []
    anomalies_par_action = {}
    
    print("📈 Collecte des données yFinance...")
    
    for symbol in SYMBOLS:
        try:
            # COLLECTE CORRIGÉE - dates explicites pour éviter les données futures
            ticker = yf.Ticker(symbol)
            
            # Utiliser des dates explicites au lieu de "period"
            start_date = "2023-01-01"
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"⚠️  Aucune donnée pour {symbol}")
                continue
            
            # Vérification des dates collectées
            dates_collectees = df.index.strftime('%Y-%m-%d').tolist()
            print(f"📅 {symbol}: {min(dates_collectees)} → {max(dates_collectees)} ({len(df)} jours)")
            
            # Préparation des données
            df = df.reset_index()
            df['symbol'] = symbol
            df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
            
            # FILTRE DE SÉCURITÉ - supprimer les dates futures
            today = datetime.now().strftime('%Y-%m-%d')
            df = df[df['date'] <= today]
            
            if df.empty:
                print(f"⚠️  Aucune donnée valide après filtrage pour {symbol}")
                continue
            
            # Nettoyage
            df_clean = clean_data(df, symbol)
            
            # Features
            df_features = create_essential_features(df_clean.copy(), symbol)
            
            # Détection des anomalies
            print(f"🔍 Analyse des anomalies pour {symbol}...")
            anomalies = detect_anomalies_avancee(df_features, symbol)
            anomalies_par_action[symbol] = anomalies
            
            if anomalies:
                print(f"   ⚠️  {len(anomalies)} anomalies détectées")
            else:
                print(f"   ✅ Aucune anomalie détectée")
            
            # Sauvegarde des 3 versions
            save_all_versions(df_clean, df_features, symbol, all_cleaned_data, all_features_data)
            
        except Exception as e:
            print(f"❌ Erreur avec {symbol}: {e}")
    
    # Sauvegarde des fichiers combinés
    if all_cleaned_data:
        combined_cleaned, combined_features = save_combined_files(all_cleaned_data, all_features_data)
        
        # Générer le rapport
        generate_simple_report(combined_cleaned, combined_features)
        
        # Sauvegarde des anomalies
        sauvegarder_anomalies(anomalies_par_action)
        
        print(f"✅ Collecte terminée")
        print(f"   • Fichiers individuels: {len([s for s in SYMBOLS if os.path.exists(f'{DATA_FOLDER}/{s}.csv')])} actions")
        print(f"   • Fichier combiné nettoyé: {len(combined_cleaned)} lignes")
        print(f"   • Fichier combiné avec features: {len(combined_features)} lignes")
        
        return combined_cleaned, combined_features
    else:
        raise Exception("Aucune donnée collectée")

# ====================== GIT ACTIONS ======================
def git_actions():
    """Gère les actions Git"""
    print("🔄 Actions Git...")
    
    subprocess.run(["git", "config", "user.email", "github-actions@github.com"])
    subprocess.run(["git", "config", "user.name", "GitHub Actions Bot"])
    
    subprocess.run(["git", "add", DATA_FOLDER])
    
    commit_msg = f"Update données + anomalies {datetime.now().strftime('%Y-%m-%d %H:%M')}"
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
    print("🚀 DÉBUT PIPELINE COMPLÈTE AVEC ANOMALIES")
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
        print("   3. RAPPORTS ANOMALIES:")
        print(f"      • ANOMALIES_DETECTEES.csv")
        print(f"      • resume_anomalies.json")
        print("   4. RAPPORT:")
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
