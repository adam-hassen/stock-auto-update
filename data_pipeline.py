import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
import json
from scipy import stats

warnings.filterwarnings('ignore')

# ====================== CONFIGURATION ======================
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

SYMBOLS = ["AAPL", "TSLA", "MSFT", "BTC-USD", "GOOGL", "NVDA", "AMZN", "META"]

# ====================== FONCTIONS OPTIMISÉES ======================
def calculate_rsi_simple(prices, period=14):
    """RSI rapide et efficace"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def clean_data_fast(df, symbol):
    """Nettoyage ultra-rapide"""
    df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['symbol'] = symbol
    
    df = df.dropna(subset=['Close'])
    df = df[df['Close'] > 0]
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def add_features(df):
    """Ajoute les features ESSENTIELLES seulement"""
    df['daily_return'] = df['Close'].pct_change()
    df['volatility_10d'] = df['daily_return'].rolling(10).std()
    
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    
    df['RSI_14'] = calculate_rsi_simple(df['Close'])
    
    df['volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_MA_5']
    
    df['target_3d'] = (df['Close'].shift(-3) / df['Close']) - 1
    
    df = df.dropna().reset_index(drop=True)
    
    return df

def detect_anomalies_essentielles(df, symbol):
    """Detection focalisee sur les signaux importants"""
    anomalies = []
    
    if len(df) < 20:
        return anomalies
    
    # 1. MOUVEMENTS EXTRÊMES
    returns = df['daily_return'].dropna()
    if len(returns) > 10:
        z_returns = np.abs(stats.zscore(returns.fillna(0)))
        extreme_idx = np.where(z_returns > 3)[0]
        
        for idx in extreme_idx:
            if idx < len(df):
                anomalies.append({
                    'symbol': symbol,
                    'date': df.iloc[idx]['date'],
                    'type': 'MOUVEMENT_EXTREME',
                    'severite': 'HAUTE',
                    'valeur': float(returns.iloc[idx] * 100),
                    'prix': float(df.iloc[idx]['Close'])
                })
    
    # 2. VOLUME ANORMAL
    volume_z = np.abs(stats.zscore(df['Volume'].fillna(0)))
    volume_idx = np.where(volume_z > 2.5)[0]
    
    for idx in volume_idx:
        if idx < len(df):
            anomalies.append({
                'symbol': symbol,
                'date': df.iloc[idx]['date'],
                'type': 'VOLUME_ANORMAL',
                'severite': 'MOYENNE',
                'volume': int(df.iloc[idx]['Volume']),
                'ratio': float(df.iloc[idx]['volume_ratio'])
            })
    
    # 3. RSI EXTRÊME
    if 'RSI_14' in df.columns:
        rsi_extreme = df[(df['RSI_14'] > 75) | (df['RSI_14'] < 25)]
        for _, row in rsi_extreme.iterrows():
            anomalies.append({
                'symbol': symbol,
                'date': row['date'],
                'type': 'RSI_EXTREME',
                'severite': 'MOYENNE',
                'rsi': float(row['RSI_14']),
                'niveau': 'SURACHAT' if row['RSI_14'] > 75 else 'SURVENDE'
            })
    
    return anomalies

def collect_and_save_all_versions():
    """Collecte et sauvegarde les 3 versions pour chaque action"""
    
    all_cleaned = []
    all_features = []
    all_anomalies = []
    
    print("COLLECTE DES DONNEES FINANCIERES")
    print("=" * 50)
    
    for i, symbol in enumerate(SYMBOLS, 1):
        try:
            print(f"\n[{i}/{len(SYMBOLS)}] Traitement de {symbol}")
            
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            df_raw = ticker.history(start=start_date, end=end_date)
            
            if df_raw.empty:
                print(f"   ERREUR: Aucune donnee pour {symbol}")
                continue
            
            df_raw = df_raw.reset_index()
            df_raw['date'] = df_raw['Date'].dt.strftime('%Y-%m-%d')
            
            df_clean = clean_data_fast(df_raw, symbol)
            
            if len(df_clean) < 30:
                print(f"   ERREUR: Pas assez de donnees apres nettoyage")
                continue
            
            df_clean.to_csv(f"{DATA_FOLDER}/{symbol}_CLEAN.csv", index=False)
            all_cleaned.append(df_clean)
            print(f"   FICHIER CREE: {symbol}_CLEAN.csv ({len(df_clean)} jours)")
            
            df_features = add_features(df_clean.copy())
            df_features.to_csv(f"{DATA_FOLDER}/{symbol}_FEATURES.csv", index=False)
            all_features.append(df_features)
            print(f"   FICHIER CREE: {symbol}_FEATURES.csv ({len(df_features)} jours, {len(df_features.columns)} colonnes)")
            
            anomalies = detect_anomalies_essentielles(df_features, symbol)
            all_anomalies.extend(anomalies)
            
            if anomalies:
                print(f"   ANOMALIES: {len(anomalies)} detectees")
            
        except Exception as e:
            print(f"   ERREUR: {str(e)[:80]}")
            continue
    
    print("\n" + "=" * 50)
    print("SAUVEGARDE DES FICHIERS COMBINES")
    
    if all_cleaned:
        df_all_clean = pd.concat(all_cleaned, ignore_index=True)
        df_all_clean.to_csv(f"{DATA_FOLDER}/ALL_CLEANED.csv", index=False)
        print(f"FICHIER CREE: ALL_CLEANED.csv ({len(df_all_clean)} lignes)")
    
    if all_features:
        df_all_features = pd.concat(all_features, ignore_index=True)
        df_all_features.to_csv(f"{DATA_FOLDER}/ALL_FEATURES.csv", index=False)
        print(f"FICHIER CREE: ALL_FEATURES.csv ({len(df_all_features)} lignes)")
    
    if all_anomalies:
        df_all_anomalies = pd.DataFrame(all_anomalies)
        df_all_anomalies.to_csv(f"{DATA_FOLDER}/ALL_ANOMALIES.csv", index=False)
        
        report = {
            'date_analyse': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_anomalies': len(all_anomalies),
            'par_type': df_all_anomalies['type'].value_counts().to_dict(),
            'par_action': df_all_anomalies['symbol'].value_counts().to_dict(),
            'dernieres_anomalies': df_all_anomalies.nlargest(10, 'date').to_dict('records')
        }
        
        with open(f"{DATA_FOLDER}/anomalies_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"FICHIER CREE: ALL_ANOMALIES.csv ({len(all_anomalies)} anomalies)")
        print(f"   REPARTITION: {report['par_type']}")
    
    print("\n" + "=" * 50)
    print("RAPPORT FINAL")
    
    summary = []
    for symbol in SYMBOLS:
        clean_file = f"{DATA_FOLDER}/{symbol}_CLEAN.csv"
        feat_file = f"{DATA_FOLDER}/{symbol}_FEATURES.csv"
        
        if os.path.exists(clean_file):
            df_c = pd.read_csv(clean_file)
            df_f = pd.read_csv(feat_file) if os.path.exists(feat_file) else None
            
            summary.append({
                'symbol': symbol,
                'status': 'OK',
                'jours_clean': len(df_c),
                'jours_features': len(df_f) if df_f is not None else 0,
                'dernier_jour': df_c['date'].iloc[-1] if len(df_c) > 0 else 'N/A',
                'dernier_prix': float(df_c['Close'].iloc[-1]) if len(df_c) > 0 else 0
            })
        else:
            summary.append({
                'symbol': symbol,
                'status': 'ECHEC',
                'jours_clean': 0,
                'jours_features': 0,
                'dernier_jour': 'N/A',
                'dernier_prix': 0
            })
    
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(f"{DATA_FOLDER}/DATA_SUMMARY.csv", index=False)
    
    print(df_summary.to_string())
    
    print("\n" + "=" * 50)
    success_count = len([s for s in summary if s['status'] == 'OK'])
    print(f"RESULTAT: {success_count}/{len(SYMBOLS)} actions collectees avec succes")
    print(f"DOSSIER DES DONNEES: {os.path.abspath(DATA_FOLDER)}")
    print("=" * 50)

# ====================== EXÉCUTION ======================
if __name__ == "__main__":
    print("LANCEMENT DE LA COLLECTE DE DONNEES FINANCIERES")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        collect_and_save_all_versions()
        print("\nCOLLECTE TERMINEE AVEC SUCCES")
    except Exception as e:
        print(f"\nERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
