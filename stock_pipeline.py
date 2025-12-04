# -*- coding: utf-8 -*-
"""
PIPELINE COMPLÈTE - Version finale avec double repo et branche spécifique
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

# BRANCHE À UTILISER - MODIFIE ICI SI BESOIN
BRANCH_NAME = "Collecte-Des-Données"  # Ta branche spécifique

# URLs des repositories - À MODIFIER
GITHUB_REPOSITORY = os.getenv('GITHUB_REPOSITORY', 'ton-username/ton-repo-principal')
REPO2_USERNAME = "Gasthorn"  # REMPLACE
REPO2_REPONAME = "Projet4A_PredictionsBoursieres"  # REMPLACE

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
    """Vérifie si le token GitHub est disponible - VERSION CORRIGÉE"""
    token = os.getenv('GITHUB_TOKEN')
    
    if not token:
        print("⚠️  GITHUB_TOKEN non trouvé dans l'environnement")
        print("   En local: export GITHUB_TOKEN=ton_token_github")
        print("   Sur GitHub Actions: configuré automatiquement")
        return False
    
    # Vérification plus simple
    if len(token) > 10:  # Un token GitHub fait au moins 40 chars, mais on simplifie
        print("✅ Token GitHub détecté")
        return True
    else:
        print("⚠️  Token GitHub semble invalide")
        return False

# ... [TOUTES LES AUTRES FONCTIONS RESTENT IDENTIQUES JUSQU'À git_actions_repo1] ...
# (Nettoyage, features, anomalies, collecte yfinance - tout reste pareil)

# ====================== GIT ACTIONS POUR REPO PRINCIPAL ======================
def git_actions_repo1():
    """Gère les actions Git pour le repo principal - VERSION AVEC BRANCHE"""
    print("\n" + "="*50)
    print(f"🔄 Actions Git - Repository principal (branche: {BRANCH_NAME})...")
    
    # Configurer Git
    subprocess.run(["git", "config", "user.email", "github-actions@github.com"], 
                   capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "GitHub Actions Bot"], 
                   capture_output=True, text=True)
    
    # Vérifier sur quelle branche on est et switcher si nécessaire
    branch_check = subprocess.run(["git", "branch", "--show-current"], 
                                capture_output=True, text=True)
    current_branch = branch_check.stdout.strip()
    
    if current_branch != BRANCH_NAME:
        print(f"🔄 Switching de '{current_branch}' à '{BRANCH_NAME}'...")
        
        # Essayer de checkout la branche si elle existe
        checkout_result = subprocess.run(["git", "checkout", BRANCH_NAME], 
                                       capture_output=True, text=True)
        
        if checkout_result.returncode != 0:
            # La branche n'existe pas, on la crée
            print(f"📝 Création de la branche '{BRANCH_NAME}'...")
            subprocess.run(["git", "checkout", "-b", BRANCH_NAME], 
                         capture_output=True, text=True)
    
    # Ajouter les fichiers data
    subprocess.run(["git", "add", DATA_FOLDER + "/*"], 
                   capture_output=True, text=True)
    
    # Vérifier s'il y a des changements
    status_result = subprocess.run(["git", "status", "--porcelain"], 
                                 capture_output=True, text=True)
    
    if not status_result.stdout.strip():
        print("✅ Aucun changement dans le repo principal")
        return False
    
    # Commit
    commit_msg = f"📊 Update données financières {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    commit_result = subprocess.run(["git", "commit", "-m", commit_msg], 
                                 capture_output=True, text=True)
    
    if commit_result.returncode != 0:
        print(f"⚠️  Erreur commit: {commit_result.stderr[:200]}")
        return False
    
    # Push vers la branche spécifique
    print(f"⬆️  Pushing vers {BRANCH_NAME}...")
    push_result = subprocess.run(["git", "push", REPO1_URL, f"HEAD:{BRANCH_NAME}", "--force"], 
                               capture_output=True, text=True)
    
    if push_result.returncode == 0:
        print(f"✅ Push réussi sur la branche {BRANCH_NAME}!")
        return True
    else:
        print(f"❌ Erreur push: {push_result.stderr[:200]}")
        return False

# ====================== PUSH VERS REPO PUBLIC ======================
def push_to_public_repo(combined_cleaned, combined_features, data_report):
    """Push les fichiers CSV vers un repository public - VERSION CORRIGÉE"""
    print("\n" + "="*50)
    print("🌐 Préparation du repository public...")
    
    # Vérifier le token directement (pas via la fonction buggée)
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ GITHUB_TOKEN non trouvé dans os.getenv('GITHUB_TOKEN')")
        print("   Vérifie que GITHUB_TOKEN est bien dans les env vars")
        return False
    
    print(f"✅ Token GitHub disponible (longueur: {len(token)})")
    
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
        
        # Ajouter le remote AVEC LE TOKEN
        print(f"🔗 Connexion au repo: {REPO2_PUBLIC_URL}")
        remote_add = subprocess.run(["git", "remote", "add", "origin", REPO2_URL], 
                                  capture_output=True, text=True)
        
        if remote_add.returncode != 0:
            print(f"⚠️  Remote déjà configuré")
        
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
            
            # Essayer une méthode alternative
            print("🔄 Essai méthode alternative...")
            push_result2 = subprocess.run(["git", "push", "-u", "origin", "main", "--force"], 
                                        capture_output=True, text=True)
            
            if push_result2.returncode == 0:
                print("✅ Push réussi avec méthode alternative!")
                return True
            else:
                print(f"❌ Échec complet: {push_result2.stderr[:200]}")
                return False
            
    except Exception as e:
        print(f"❌ Erreur lors du push vers le repo public: {e}")
        import traceback
        traceback.print_exc()
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
    print(f"📌 Branche cible: {BRANCH_NAME}")
    print(f"🌐 Repo public: {REPO2_PUBLIC_URL}")
    
    try:
        # 1. Vérifier le token GitHub (juste pour info)
        token = os.getenv('GITHUB_TOKEN')
        if token:
            print(f"✅ Token GitHub détecté")
        else:
            print("⚠️  GITHUB_TOKEN non trouvé, certaines fonctionnalités seront limitées")
        
        # 2. Collecter et traiter les données
        combined_cleaned, combined_features, data_report = collect_yfinance()
        
        # 3. Push vers le repo principal (sur la bonne branche)
        repo1_success = git_actions_repo1()
        
        # 4. Push vers le repo public (seulement si on a un token)
        repo2_success = False
        if token:
            repo2_success = push_to_public_repo(combined_cleaned, combined_features, data_report)
        else:
            print("\n" + "="*50)
            print("⚠️  Pas de token, skip du repo public")
            print("   Pour activer: configure GITHUB_TOKEN dans GitHub Actions")
        
        # 5. Résumé final
        print("\n" + "="*60)
        print("🎉 RÉSUMÉ DE L'EXÉCUTION:")
        print("-"*60)
        
        if repo1_success:
            print(f"✅ REPO PRINCIPAL: Données mises à jour sur '{BRANCH_NAME}'")
        else:
            print("ℹ️  REPO PRINCIPAL: Aucun changement détecté")
        
        if repo2_success:
            print(f"✅ REPO PUBLIC: Données publiées avec succès")
            print(f"   📍 {REPO2_PUBLIC_URL}")
        elif token:
            print("❌ REPO PUBLIC: Échec de la publication")
        else:
            print("⚠️  REPO PUBLIC: Skippé (pas de token)")
        
        print("\n📊 STATISTIQUES FINALES:")
        print(f"   • Actions traitées: {len([s for s in SYMBOLS if os.path.exists(f'{DATA_FOLDER}/{s}.csv')])}/{len(SYMBOLS)}")
        print(f"   • Données totales: {len(combined_cleaned) if combined_cleaned is not None else 0} lignes")
        print(f"   • Anomalies détectées: Voir {DATA_FOLDER}/ANOMALIES_DETECTEES.csv")
        print(f"   • Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print("🏁 PIPELINE TERMINÉE")
        
    except Exception as e:
        print(f"\n💥 ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
