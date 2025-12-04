# -*- coding: utf-8 -*-
"""
PIPELINE COMPLÈTE - Version finale avec double repo
yFinance → Nettoyage → Features → Anomalies → Git (repo principal + repo public)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import subprocess
import json
import warnings
import shutil
import tempfile
warnings.filterwarnings('ignore')

# ====================== CONFIGURATION ======================
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# URLs des repositories - À MODIFIER POUR TON REPO PUBLIC
GITHUB_REPOSITORY = os.getenv('GITHUB_REPOSITORY', 'ton-username/ton-repo-principal')
REPO2_USERNAME = "Gasthorn"  # REMPLACE PAR TON USERNAME
REPO2_REPONAME = "Projet4A_PredictionsBoursieres"  # REMPLACE PAR LE NOM DE TON REPO PUBLIC

REPO1_URL = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{GITHUB_REPOSITORY}.git"
REPO2_PUBLIC_URL = f"https://github.com/{REPO2_USERNAME}/{REPO2_REPONAME}.git"
REPO2_URL = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{REPO2_USERNAME}/{REPO2_REPONAME}.git"

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

def check_github_token():
    """Vérifie si le token GitHub est disponible"""
    if not GITHUB_TOKEN:
        print("⚠️  GITHUB_TOKEN non trouvé dans l'environnement")
        print("   En local: export GITHUB_TOKEN=ton_token_github")
        print("   Sur GitHub Actions: configuré automatiquement")
        return False
    
    # Vérification basique du format du token
    if len(GITHUB_TOKEN) < 20:
        print("⚠️  Token GitHub semble trop court")
        return False
    
    print("✅ Token GitHub détecté")
    return True

# ====================== DÉTECTION D'ANOMALIES ======================
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
                    'description': f"Mouvement de prix extrême: {rendement:.2f}%",
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
    
    print(f"✅ {len(toutes_anomalies)} anomalies détectées et sauvegardées")
    if resume_anomalies['par_type']:
        print(f"   📊 Répartition: {resume_anomalies['par_type']}")

# ====================== NETTOYAGE DES DONNÉES ======================
def clean_data(df, symbol):
    """Nettoyage des données financières"""
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
    
    print(f"✅ {symbol}: {len(df)} lignes avec features")
    return df

# ====================== SAUVEGARDE DES VERSIONS ======================
def save_all_versions(df_clean, df_features, symbol, all_cleaned_data, all_features_data):
    """Sauvegarde les 3 versions des données"""
    
    df_features.to_csv(f"{DATA_FOLDER}/{symbol}.csv", index=False)
    all_cleaned_data.append(df_clean)
    all_features_data.append(df_features)
    
    print(f"💾 {symbol}.csv sauvegardé")

def save_combined_files(all_cleaned_data, all_features_data):
    """Sauvegarde les fichiers combinés"""
    
    combined_cleaned = pd.concat(all_cleaned_data, ignore_index=True)
    combined_cleaned.to_csv(f"{DATA_FOLDER}/ALL_CLEANED.csv", index=False)
    print("💾 ALL_CLEANED.csv sauvegardé")
    
    combined_features = pd.concat(all_features_data, ignore_index=True)
    combined_features.to_csv(f"{DATA_FOLDER}/ALL_FEATURES.csv", index=False)
    print("💾 ALL_FEATURES.csv sauvegardé")
    
    return combined_cleaned, combined_features

def generate_simple_report(combined_cleaned):
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
                'last_volume': int(symbol_data['Volume'].iloc[-1]),
                'update_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(f"{DATA_FOLDER}/data_report.csv", index=False)
        print("✅ Rapport généré")
    
    return report_df

# ====================== COLLECTE YFINANCE ======================
def collect_yfinance():
    """Collecte principale des données yFinance"""
    all_cleaned_data = []
    all_features_data = []
    anomalies_par_action = {}
    
    print("📈 Collecte des données yFinance...")
    print(f"📅 Date du jour: {datetime.now().strftime('%Y-%m-%d')}")
    
    for symbol in SYMBOLS:
        try:
            print(f"\n🔍 Récupération {symbol}...")
            ticker = yf.Ticker(symbol)
            
            # Récupérer les 180 derniers jours
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"⚠️  Aucune donnée pour {symbol}")
                continue
            
            dates_collectees = df.index.strftime('%Y-%m-%d').tolist()
            print(f"   📅 Période: {min(dates_collectees)} → {max(dates_collectees)} ({len(df)} jours)")
            
            df = df.reset_index()
            df['symbol'] = symbol
            df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
            
            today = datetime.now().strftime('%Y-%m-%d')
            df = df[df['date'] <= today]
            
            if df.empty:
                print(f"⚠️  Aucune donnée valide après filtrage pour {symbol}")
                continue
            
            df_clean = clean_data(df, symbol)
            df_features = create_essential_features(df_clean.copy(), symbol)
            
            print(f"   🔍 Analyse des anomalies...")
            anomalies = detect_anomalies_avancee(df_features, symbol)
            anomalies_par_action[symbol] = anomalies
            
            if anomalies:
                print(f"   ⚠️  {len(anomalies)} anomalies détectées")
            else:
                print(f"   ✅ Aucune anomalie")
            
            save_all_versions(df_clean, df_features, symbol, all_cleaned_data, all_features_data)
            
        except Exception as e:
            print(f"❌ Erreur avec {symbol}: {str(e)[:100]}...")
    
    if all_cleaned_data:
        combined_cleaned, combined_features = save_combined_files(all_cleaned_data, all_features_data)
        data_report = generate_simple_report(combined_cleaned)
        sauvegarder_anomalies(anomalies_par_action)
        
        print(f"\n✅ Collecte terminée avec succès!")
        print(f"   • {len([s for s in SYMBOLS if os.path.exists(f'{DATA_FOLDER}/{s}.csv')])}/{len(SYMBOLS)} actions récupérées")
        print(f"   • ALL_CLEANED.csv: {len(combined_cleaned)} lignes")
        print(f"   • ALL_FEATURES.csv: {len(combined_features)} lignes")
        
        return combined_cleaned, combined_features, data_report
    else:
        raise Exception("❌ Aucune donnée collectée")

# ====================== GIT ACTIONS POUR REPO PRINCIPAL ======================
def git_actions_repo1():
    """Gère les actions Git pour le repo principal"""
    print("\n" + "="*50)
    print("🔄 Actions Git - Repository principal...")
    
    # Configurer Git
    subprocess.run(["git", "config", "user.email", "github-actions@github.com"], 
                   capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "GitHub Actions Bot"], 
                   capture_output=True, text=True)
    
    # Ajouter les fichiers data
    subprocess.run(["git", "add", DATA_FOLDER + "/*"], 
                   capture_output=True, text=True)
    
    # Commit
    commit_msg = f"📊 Update données financières {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    commit_result = subprocess.run(["git", "commit", "-m", commit_msg], 
                                 capture_output=True, text=True)
    
    if commit_result.returncode != 0 or "nothing to commit" in commit_result.stdout:
        print("✅ Aucun changement dans le repo principal")
        return False
    
    # Push vers le repo principal
    print("⬆️  Pushing vers le repo principal...")
    push_result = subprocess.run(["git", "push", REPO1_URL, "HEAD:main"], 
                               capture_output=True, text=True)
    
    if push_result.returncode == 0:
        print("✅ Push réussi sur le repo principal!")
        return True
    else:
        print(f"❌ Erreur push repo principal: {push_result.stderr[:200]}")
        return False

# ====================== PUSH VERS REPO PUBLIC ======================
def push_to_public_repo(combined_cleaned, combined_features, data_report):
    """Push les fichiers CSV vers un repository public"""
    print("\n" + "="*50)
    print("🌐 Préparation du repository public...")
    
    if not check_github_token():
        print("❌ Impossible de push sans token GitHub")
        return False
    
    temp_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    
    try:
        print(f"📁 Dossier temporaire: {temp_dir}")
        os.chdir(temp_dir)
        
        # Initialiser un nouveau repo git
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "data-pipeline@github.com"], check=False)
        subprocess.run(["git", "config", "user.name", "Financial Data Bot"], check=False)
        
        # Créer la structure
        os.makedirs("data", exist_ok=True)
        
        # Sauvegarder les fichiers principaux
        print("💾 Sauvegarde des fichiers dans le repo public...")
        
        if combined_cleaned is not None and not combined_cleaned.empty:
            combined_cleaned.to_csv("data/ALL_CLEANED.csv", index=False)
            print(f"   • ALL_CLEANED.csv: {len(combined_cleaned)} lignes")
        
        if combined_features is not None and not combined_features.empty:
            combined_features.to_csv("data/ALL_FEATURES.csv", index=False)
            print(f"   • ALL_FEATURES.csv: {len(combined_features)} lignes")
        
        if data_report is not None and not data_report.empty:
            data_report.to_csv("data/data_report.csv", index=False)
            print(f"   • data_report.csv: {len(data_report)} actions")
        
        # Ajouter un README
        readme_content = f"""# 📊 Archive de Données Financières

*Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## 📁 Fichiers disponibles

| Fichier | Description | Taille |
|---------|-------------|--------|
| `data/ALL_CLEANED.csv` | Données brutes nettoyées | {len(combined_cleaned) if combined_cleaned is not None else 0} lignes |
| `data/ALL_FEATURES.csv` | Données avec indicateurs techniques | {len(combined_features) if combined_features is not None else 0} lignes |
| `data/data_report.csv` | Rapport des données disponibles | {len(data_report) if data_report is not None else 0} actions |

## 📈 Symboles suivis

{', '.join(SYMBOLS)}

## 🔍 Source des données

Données collectées depuis Yahoo Finance via l'API yFinance.
Mises à jour automatiques quotidiennes.

## 📄 Structure des données

Chaque fichier CSV contient les colonnes suivantes:
- `date`: Date de la donnée (AAAA-MM-JJ)
- `symbol`: Symbole de l'action/crypto
- `Open`, `High`, `Low`, `Close`: Prix d'ouverture, plus haut, plus bas, clôture
- `Volume`: Volume échangé
- `daily_return`, `volatility_10`, `MA_*`, `RSI_14`: Indicateurs techniques

## ⚠️ Avertissement

Ces données sont fournies à titre informatif uniquement.
Ne constitue pas un conseil en investissement.

---

*Généré automatiquement par [GitHub Actions](https://github.com/{GITHUB_REPOSITORY})*
"""
        
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # Ajouter un .gitignore
        with open(".gitignore", "w") as f:
            f.write("*.pyc\n__pycache__/\n*.log\n.DS_Store\n")
        
        # Ajouter tous les fichiers
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        
        # Commit
        commit_msg = f"📈 Update données {datetime.now().strftime('%Y-%m-%d')}"
        commit_result = subprocess.run(["git", "commit", "-m", commit_msg], 
                                     capture_output=True, text=True)
        
        if commit_result.returncode != 0 and "nothing to commit" not in commit_result.stdout:
            print(f"❌ Erreur commit: {commit_result.stderr}")
            return False
        
        # Créer la branche main
        subprocess.run(["git", "branch", "-M", "main"], check=True, capture_output=True)
        
        # Ajouter le remote
        print(f"🔗 Connexion au repo: {REPO2_PUBLIC_URL}")
        subprocess.run(["git", "remote", "add", "origin", REPO2_URL], 
                      capture_output=True, text=True)
        
        # Force push (on écrase tout car c'est juste des données)
        print("⬆️  Pushing vers le repo public...")
        push_result = subprocess.run(["git", "push", "--force", "origin", "main"], 
                                   capture_output=True, text=True)
        
        if push_result.returncode == 0:
            print(f"✅ Push réussi sur le repo public!")
            print(f"🔗 URL: {REPO2_PUBLIC_URL}")
            return True
        else:
            print(f"❌ Erreur push: {push_result.stderr[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du push vers le repo public: {e}")
        return False
    finally:
        os.chdir(original_dir)
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

# ====================== MAIN ======================
if __name__ == "__main__":
    print("🚀 DÉBUT PIPELINE COMPLÈTE - DOUBLE REPOSITORY")
    print("="*60)
    
    try:
        # 1. Vérifier le token GitHub
        check_github_token()
        
        # 2. Collecter et traiter les données
        combined_cleaned, combined_features, data_report = collect_yfinance()
        
        # 3. Push vers le repo principal
        repo1_success = git_actions_repo1()
        
        # 4. Push vers le repo public
        repo2_success = push_to_public_repo(combined_cleaned, combined_features, data_report)
        
        # 5. Résumé final
        print("\n" + "="*60)
        print("🎉 RÉSUMÉ DE L'EXÉCUTION:")
        print("-"*60)
        
        if repo1_success:
            print("✅ REPO PRINCIPAL: Données mises à jour avec succès")
        else:
            print("ℹ️  REPO PRINCIPAL: Aucun changement détecté")
        
        if repo2_success:
            print(f"✅ REPO PUBLIC: Données publiées avec succès")
            print(f"   📍 {REPO2_PUBLIC_URL}")
        else:
            print("❌ REPO PUBLIC: Échec de la publication")
        
        print("\n📊 STATISTIQUES FINALES:")
        print(f"   • Actions traitées: {len([s for s in SYMBOLS if os.path.exists(f'{DATA_FOLDER}/{s}.csv')])}/{len(SYMBOLS)}")
        print(f"   • Données totales: {len(combined_cleaned) if combined_cleaned is not None else 0} lignes")
        print(f"   • Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print("🏁 PIPELINE TERMINÉE")
        
    except Exception as e:
        print(f"\n💥 ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
