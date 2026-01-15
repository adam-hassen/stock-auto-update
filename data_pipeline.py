import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
import json
warnings.filterwarnings('ignore')

# ====================== CONFIGURATION ======================
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Symboles à tracker
SYMBOLS = ["AAPL", "TSLA", "MSFT", "BTC-USD", "GOOGL", "NVDA", "AMZN", "META"]

# ====================== FONCTIONS UTILITAIRES ======================
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
    """Detection avancee des anomalies dans les donnees financieres"""
    
    anomalies = []
    
    # 1. ANOMALIES DE PRIX
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
                    'description': f"Mouvement de prix extreme: {rendement:.2f}%",
                    'prix_ce_jour': prix
                })
    
    # 2. ANOMALIES DE VOLUME
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
    
    # 3. GAPS DE PRIX
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
    
    # 4. ANOMALIES DE VOLATILITE
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
                        'description': f"Volatilite extreme: {volatilite:.2f}%",
                        'periode': '10 jours'
                    })
    
    # 5. CASSURES DE TENDANCE
    for window in [20, 50]:
        ma_col = f'MA_{window}'
        if ma_col in df.columns:
            cassure_hausse = (df['Close'] > df[ma_col]) & (df['Close'].shift(1) <= df[ma_col].shift(1))
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
    
    # 6. RSI EXTREME
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
    """Sauvegarde toutes les anomalies detectees"""
    if not any(anomalies_par_action.values()):
        print("Aucune anomalie detectee")
        return
    
    toutes_anomalies = []
    for symbol, anomalies in anomalies_par_action.items():
        toutes_anomalies.extend(anomalies)
    
    df_anomalies = pd.DataFrame(toutes_anomalies)
    df_anomalies.to_csv(f"{DATA_FOLDER}/ANOMALIES_DETECTEES.csv", index=False)
    
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
    
    with open(f"{DATA_FOLDER}/resume_anomalies.json", "w") as f:
        json.dump(resume_anomalies, f, indent=2, ensure_ascii=False)
    
    print(f"{len(toutes_anomalies)} anomalies detectees et sauvegardees")
    if resume_anomalies['par_type']:
        print(f"    Repartition: {resume_anomalies['par_type']}")

# ====================== NETTOYAGE DES DONNEES ======================
def clean_data(df, symbol):
    """Nettoyage des donnees financieres"""
    df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    df = df[df['Close'] > 0]
    df = df.dropna(subset=['Close'])
    
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['symbol'] = symbol
    
    return df

# ====================== FEATURES ESSENTIELLES ======================
def create_essential_features(df, symbol):
    """Features les plus importantes seulement"""
    df = df.sort_values('date').reset_index(drop=True)
    
    df['daily_return'] = df['Close'].pct_change()
    df['volatility_10'] = df['daily_return'].rolling(10).std()
    
    for window in [5, 20, 50]:
        df[f'MA_{window}'] = df['Close'].rolling(window).mean()
    
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI_14'] = calculate_rsi(df['Close'])
    
    df['volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_MA_5']
    
    df['target_3_days'] = (df['Close'].shift(-3) / df['Close']) - 1
    df['target_direction'] = (df['target_3_days'] > 0).astype(int)
    
    df = df.dropna()
    
    print(f"{symbol}: {len(df)} lignes avec features")
    return df

# ====================== SAUVEGARDE DES VERSIONS ======================
def save_all_versions(df_clean, df_features, symbol, all_cleaned_data, all_features_data):
    """Sauvegarde les 3 versions des donnees"""
    
    df_features.to_csv(f"{DATA_FOLDER}/{symbol}.csv", index=False)
    all_cleaned_data.append(df_clean)
    all_features_data.append(df_features)
    
    print(f"{symbol}.csv sauvegarde")

def save_combined_files(all_cleaned_data, all_features_data):
    """Sauvegarde les fichiers combines"""
    
    combined_cleaned = pd.concat(all_cleaned_data, ignore_index=True)
    combined_cleaned.to_csv(f"{DATA_FOLDER}/ALL_CLEANED.csv", index=False)
    print("ALL_CLEANED.csv sauvegarde")
    
    combined_features = pd.concat(all_features_data, ignore_index=True)
    combined_features.to_csv(f"{DATA_FOLDER}/ALL_FEATURES.csv", index=False)
    print("ALL_FEATURES.csv sauvegarde")
    
    return combined_cleaned, combined_features

def generate_simple_report(combined_cleaned):
    """Rapport simple des donnees disponibles"""
    print("Generation du rapport...")
    
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
                'last_volume': int(symbol_data['Volume'].iloc[-1]),
                'update_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(f"{DATA_FOLDER}/data_report.csv", index=False)
        print("Rapport genere")
    
    return report_df

# ====================== COLLECTE YFINANCE ======================
def collect_yfinance():
    """Collecte principale des donnees yFinance"""
    all_cleaned_data = []
    all_features_data = []
    anomalies_par_action = {}
    
    print("Collecte des donnees yFinance...")
    print(f"Date du jour: {datetime.now().strftime('%Y-%m-%d')}")
    
    for symbol in SYMBOLS:
        try:
            print(f"Recuperation {symbol}...")
            ticker = yf.Ticker(symbol)
            
            start_date = "2023-01-01"
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            try:
                df = ticker.history(period="max")
                print(f"    Recuperation MAXIMUM de donnees")
            except:
                df = ticker.history(start=start_date, end=end_date)
                print(f"    Recuperation depuis {start_date}")
            
            if df.empty:
                print(f"Aucune donnee pour {symbol}")
                continue
            
            dates_collectees = df.index.strftime('%Y-%m-%d').tolist()
            print(f"    Periode: {min(dates_collectees)} -> {max(dates_collectees)} ({len(df)} jours)")
            
            df = df.reset_index()
            df['symbol'] = symbol
            df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
            
            today = datetime.now().strftime('%Y-%m-%d')
            df = df[df['date'] <= today]
            
            if df.empty:
                print(f"  Aucune donnee valide apres filtrage pour {symbol}")
                continue
            
            df_clean = clean_data(df, symbol)
            
            print(f"     Creation des features...")
            
            cutoff_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
            df_recent = df_clean.copy()
            df_recent = df_recent[df_recent['date'] >= cutoff_date]
            
            df_features = create_essential_features(df_recent, symbol)
            
            print(f"   Analyse des anomalies...")
            anomalies = detect_anomalies_avancee(df_features, symbol)
            anomalies_par_action[symbol] = anomalies
            
            if anomalies:
                print(f"   {len(anomalies)} anomalies detectees")
            else:
                print(f"   Aucune anomalie")
            
            df_features.to_csv(f"{DATA_FOLDER}/{symbol}.csv", index=False)
            all_cleaned_data.append(df_clean)
            all_features_data.append(df_features)
            
            print(f"{symbol}.csv sauvegarde ({len(df_clean)} lignes brutes, {len(df_features)} lignes avec features)")
            
        except Exception as e:
            print(f"Erreur avec {symbol}: {str(e)[:100]}...")
    
    if all_cleaned_data:
        combined_cleaned, combined_features = save_combined_files(all_cleaned_data, all_features_data)
        data_report = generate_simple_report(combined_cleaned)
        sauvegarder_anomalies(anomalies_par_action)
        
        print(f"Collecte terminee avec succes!")
        print(f"   {len([s for s in SYMBOLS if os.path.exists(f'{DATA_FOLDER}/{s}.csv')])}/{len(SYMBOLS)} actions recuperees")
        print(f"   ALL_CLEANED.csv: {len(combined_cleaned)} lignes")
        print(f"   ALL_FEATURES.csv: {len(combined_features)} lignes")
        
        return combined_cleaned, combined_features, data_report
    else:
        raise Exception("Aucune donnee collectee")

# ====================== MAIN ======================
if __name__ == "__main__":
    print("DEBUT PIPELINE COMPLETE")
    print("="*60)
    print(f"Symboles analyses: {', '.join(SYMBOLS)}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        combined_cleaned, combined_features, data_report = collect_yfinance()
        
        print("="*60)
        print("STATISTIQUES FINALES:")
        print(f"   Actions traitees: {len([s for s in SYMBOLS if os.path.exists(f'{DATA_FOLDER}/{s}.csv')])}/{len(SYMBOLS)}")
        print(f"   Donnees totales: {len(combined_cleaned) if combined_cleaned is not None else 0} lignes")
        print(f"   Anomalies detectees: {DATA_FOLDER}/ANOMALIES_DETECTEES.csv")
        print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print("PIPELINE TERMINEE")
        
    except Exception as e:
        print(f"ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
