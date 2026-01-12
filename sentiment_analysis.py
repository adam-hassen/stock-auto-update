# ==================== IMPORT DES LIBRAIRIES ====================
import requests
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
import subprocess
import tempfile
import shutil
import glob
import json
from typing import List, Dict, Any
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    """Configuration centrale du projet"""
    
    API_SOURCE = 'newsapi'
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', "ed172154b55e4d4eb95db4ac7895b29e")
    MAX_ARTICLES = 15
    DAYS_BACK = 3
    
    # Configuration GitHub
    PUSH_TOKEN = os.getenv('PUSH_TOKEN')
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', PUSH_TOKEN)
    
    # Repository pour les donnees sentiment
    REPO_OWNER = "Gasthorn"
    REPO_NAME = "Projet4A_PredictionsBoursieres"
    REPO_BRANCH = "Collecte-Des-Donnees"
    
    # Dossier pour sauvegarder les donnees
    SENTIMENT_FOLDER = "sentiment_data"
    os.makedirs(SENTIMENT_FOLDER, exist_ok=True)

# ==================== FONCTIONS GIT ====================
def git_push_to_repo():
    """Pousse les fichiers CSV vers le repo GitHub"""
    
    if not Config.PUSH_TOKEN:
        print("PUSH_TOKEN non configure. Impossible de pousser vers GitHub.")
        return False
    
    print("\n" + "="*60)
    print("Preparation du push vers GitHub...")
    print(f"Repo: {Config.REPO_OWNER}/{Config.REPO_NAME}")
    print(f"Branche: {Config.REPO_BRANCH}")
    print("="*60)
    
    # Chemin vers les fichiers CSV générés
    original_dir = os.getcwd()
    
    # Vérifier si les fichiers existent dans sentiment_data/
    csv_files = glob.glob(f"{original_dir}/{Config.SENTIMENT_FOLDER}/articles_*.csv")
    
    if not csv_files:
        # Essayer aussi à la racine
        csv_files = glob.glob(f"{original_dir}/articles_*.csv")
    
    if not csv_files:
        print("Aucun fichier CSV trouve a pousser")
        print(f"Recherche dans: {original_dir}/{Config.SENTIMENT_FOLDER}/")
        print(f"Recherche dans: {original_dir}/")
        return False
    
    print(f"Fichiers a pousser: {len(csv_files)}")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
    
    # Creer un dossier temporaire
    temp_dir = tempfile.mkdtemp()
    
    try:
        print(f"\nDossier temporaire: {temp_dir}")
        os.chdir(temp_dir)
        
        # Initialiser un nouveau repo git
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "sentiment-bot@github.com"], check=False)
        subprocess.run(["git", "config", "user.name", "Sentiment Analysis Bot"], check=False)
        
        # Creer la structure de dossiers
        os.makedirs("sentiment", exist_ok=True)
        
        # Copier chaque fichier CSV
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            dest_path = os.path.join("sentiment", filename)
            shutil.copy2(csv_file, dest_path)
            print(f"  Copie: sentiment/{filename}")
        
        # Créer un README spécifique
        readme_content = f"""# Donnees d'Analyse de Sentiment

*Derniere mise a jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Description

Ce dossier contient les articles financiers pour differentes entreprises.
Les articles sont recuperes via NewsAPI.

## Entreprises analysees

- Apple (AAPL)
- Microsoft (MSFT)  
- Tesla (TSLA)
- Nvidia (NVDA)

## Fichiers disponibles

Chaque fichier contient les articles pour une entreprise specifique.
Colonnes principales:

| Colonne | Description |
|---------|-------------|
| date | Date de publication de l'article |
| source | Source de l'article (Reuters, Bloomberg, etc.) |
| titre | Titre de l'article |
| contenu | Contenu de l'article |
| score | Score de sentiment (-1 a +1) |
| sentiment | Categorie (positive/negative/neutral) |

---

*Genere automatiquement par GitHub Actions*
"""

        with open("sentiment/README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # Ajouter un .gitignore
        with open(".gitignore", "w") as f:
            f.write("*.pyc\n__pycache__/\n*.log\n.DS_Store\n")
        
        # Ajouter tous les fichiers
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        
        # Commit
        commit_msg = f"Mise a jour articles {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        commit_result = subprocess.run(["git", "commit", "-m", commit_msg], 
                                     capture_output=True, text=True)
        
        if commit_result.returncode != 0 and "nothing to commit" not in commit_result.stdout:
            print(f"Erreur commit: {commit_result.stderr[:200]}")
            # Essayer de voir ce qui se passe
            subprocess.run(["git", "status"], capture_output=True, text=True)
            return False
        
        # Créer la branche spécifiée
        subprocess.run(["git", "branch", "-M", Config.REPO_BRANCH], check=True, capture_output=True)
        
        # URL du repo avec token
        repo_url = f"https://x-access-token:{Config.PUSH_TOKEN}@github.com/{Config.REPO_OWNER}/{Config.REPO_NAME}.git"
        
        # Ajouter le remote
        remote_add = subprocess.run(["git", "remote", "add", "origin", repo_url], 
                                  capture_output=True, text=True)
        
        if remote_add.returncode != 0:
            print("Remote deja configure")
        
        # Force push vers la branche
        print(f"\nPushing vers la branche {Config.REPO_BRANCH}...")
        push_result = subprocess.run(["git", "push", "--force", "origin", Config.REPO_BRANCH], 
                                   capture_output=True, text=True)
        
        if push_result.returncode == 0:
            print(f"Push reussi sur la branche {Config.REPO_BRANCH}!")
            print(f"URL: https://github.com/{Config.REPO_OWNER}/{Config.REPO_NAME}/tree/{Config.REPO_BRANCH}/sentiment")
            return True
        else:
            print(f"Erreur push: {push_result.stderr[:200]}")
            return False
            
    except Exception as e:
        print(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

# ==================== CORE API FUNCTIONS ====================
class NewsAPIClient:
    """Client pour recuperer les articles"""

    @staticmethod
    def get_news_api_articles(query: str, api_key: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Recupere les articles via NewsAPI"""
        from_date = (datetime.now() - timedelta(days=Config.DAYS_BACK)).strftime('%Y-%m-%d')

        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f'{query} AND (stock OR shares OR earnings OR revenue OR market)',
            'apiKey': api_key,
            'pageSize': max_results,
            'language': 'en',
            'sortBy': 'relevancy',
            'from': from_date,
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json().get('articles', [])
            else:
                print(f"NewsAPI Erreur {response.status_code}")
                return []
        except Exception as e:
            print(f"Erreur connexion: {e}")
            return []

# ==================== ANALYSE SIMPLE ====================
class SimpleSentimentAnalyzer:
    """Analyseur de sentiment simple sans transformers"""
    
    def __init__(self):
        print("Analyseur simple initialise (sans FinBERT)")
        # Liste de mots positifs et negatifs pour analyse basique
        self.positive_words = [
            'profit', 'gain', 'growth', 'increase', 'rise', 'up', 'positive',
            'strong', 'beat', 'success', 'win', 'bullish', 'optimistic', 'good',
            'profit', 'gains', 'growing', 'increasing', 'rising', 'positive',
            'stronger', 'beats', 'successful', 'wins', 'bullish', 'optimism'
        ]
        
        self.negative_words = [
            'loss', 'decline', 'decrease', 'fall', 'down', 'negative',
            'weak', 'miss', 'fail', 'bearish', 'pessimistic', 'bad', 'drop',
            'losses', 'declining', 'decreasing', 'falling', 'negative',
            'weaker', 'misses', 'failure', 'bearish', 'pessimism'
        ]
    
    def build_query(self, ticker: str, company_name: str) -> str:
        """Construit une requete pour l'entreprise"""
        return f'"{company_name}" OR {ticker}'
    
    def simple_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Analyse sentiment simple basee sur les mots cles"""
        try:
            text_lower = str(text).lower().strip()
            
            if len(text_lower) < 20:
                return {'score': 0, 'label': 'neutral', 'confidence': 0.1}
            
            # Compter les mots positifs et negatifs
            positive_count = sum(1 for word in self.positive_words if word in text_lower)
            negative_count = sum(1 for word in self.negative_words if word in text_lower)
            
            # Calculer le score simple
            total_words = positive_count + negative_count
            if total_words == 0:
                return {'score': 0, 'label': 'neutral', 'confidence': 0.1}
            
            score = (positive_count - negative_count) / total_words
            score = max(-1.0, min(1.0, score))
            
            # Determiner le label
            if score > 0.1:
                label = 'positive'
            elif score < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Calculer la confiance (simple)
            confidence = min(1.0, total_words / 10)  # Plus de mots trouves = plus de confiance
            
            return {
                'score': score,
                'label': label,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Erreur analyse simple: {e}")
            return {'score': 0, 'label': 'neutral', 'confidence': 0.1}
    
    def calculate_weight(self, article: Dict[str, Any]) -> float:
        """Calcule l'importance de l'article"""
        weight = 1.0

        # 1. Actualite
        if 'publishedAt' in article and article['publishedAt']:
            try:
                pub_str = str(article['publishedAt'])
                if 'T' in pub_str:
                    pub_date = datetime.strptime(pub_str, '%Y-%m-%dT%H:%M:%SZ')
                else:
                    pub_date = datetime.strptime(pub_str, '%Y-%m-%d %H:%M:%S')

                hours_old = (datetime.now() - pub_date).total_seconds() / 3600
                if hours_old <= 24:
                    weight *= 1.3
                elif hours_old <= 48:
                    weight *= 1.1
                else:
                    weight *= 0.9
            except:
                pass

        # 2. Source
        source_name = article.get('source', {}).get('name', '').lower()
        
        # Sources importantes
        important_sources = ['reuters', 'bloomberg', 'cnbc', 'financial times', 'wall street journal']
        if any(source in source_name for source in important_sources):
            weight *= 1.5
        # Blogs personnels
        elif any(blog in source_name for blog in ['blog', 'personal', 'medium']):
            weight *= 0.6

        # 3. Longueur
        title = str(article.get('title', ''))
        description = str(article.get('description', ''))
        content_length = len(title + description)

        if content_length > 300:
            weight *= 1.2
        elif content_length < 100:
            weight *= 0.8

        # Limites
        return max(0.2, min(weight, 3.0))
    
    def analyze_company(self, ticker: str, company_name: str) -> pd.DataFrame:
        """Analyse les articles d'une entreprise"""
        print(f"\n{'='*60}")
        print(f"ANALYSE POUR {company_name} ({ticker})")
        print(f"{'='*60}")

        # Recuperation des articles
        articles = NewsAPIClient.get_news_api_articles(
            self.build_query(ticker, company_name),
            Config.NEWSAPI_KEY,
            Config.MAX_ARTICLES
        )

        if not articles:
            print("Aucun article trouve")
            return pd.DataFrame()

        analyzed_articles = []
        
        print(f"\n{len(articles)} articles trouves:")
        print("-" * 60)

        for i, article in enumerate(articles, 1):
            # Texte complet
            title = str(article.get('title', 'Sans titre')).strip()
            description = str(article.get('description', '')).strip()
            content = f"{title}. {description}"

            # Analyse sentiment simple
            sentiment = self.simple_sentiment_analysis(content)
            
            # Calcul poids
            weight = self.calculate_weight(article)
            
            # Score pondere
            weighted_score = sentiment['score'] * weight

            # Date
            pub_date = "Inconnue"
            if 'publishedAt' in article and article['publishedAt']:
                try:
                    date_str = str(article['publishedAt'])
                    if 'T' in date_str:
                        pub_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M')
                    else:
                        pub_date = date_str[:16]
                except:
                    pub_date = date_str[:10] if len(date_str) >= 10 else "Inconnue"

            # Source
            source_name = article.get('source', {}).get('name', 'Inconnue')

            # Article complet
            analyzed_article = {
                'id': i,
                'date': pub_date,
                'source': str(source_name),
                'titre': title[:100] + '...' if len(title) > 100 else title,
                'contenu': content[:200] + '...' if len(content) > 200 else content,
                'score': round(sentiment['score'], 3),
                'sentiment': sentiment['label'],
                'confiance': round(sentiment['confidence'], 3),
                'poids': round(weight, 2),
                'score_pondere': round(weighted_score, 3),
                'url': article.get('url', '')
            }

            analyzed_articles.append(analyzed_article)

            # Affichage simple
            print(f"Article {i}: {title[:50]}...")
            print(f"  Score: {sentiment['score']:.3f} | Sentiment: {sentiment['label']} | Source: {source_name}")

        # Creation du DataFrame
        df = pd.DataFrame(analyzed_articles)

        # Sauvegarde CSV dans le dossier sentiment_data
        if not df.empty:
            filename = f"{Config.SENTIMENT_FOLDER}/articles_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"\nDonnees sauvegardees: {filename}")
            
            # Statistiques simples
            positif = len(df[df['score'] > 0.1])
            negatif = len(df[df['score'] < -0.1])
            neutre = len(df) - positif - negatif
            
            print(f"\nRESUME:")
            print(f"  Articles positifs: {positif}")
            print(f"  Articles negatifs: {negatif}")
            print(f"  Articles neutres: {neutre}")
            print(f"  Score moyen: {df['score'].mean():.3f}")

        return df

# ==================== EXECUTION PRINCIPALE ====================
def main():
    """Fonction principale"""
    print("DEBUT DE L'ANALYSE D'ARTICLES")
    print(f"Source: {Config.API_SOURCE}")
    print(f"Periode: {Config.DAYS_BACK} jours")
    print("=" * 60)

    # Initialisation
    analyzer = SimpleSentimentAnalyzer()

    # Entreprises a analyser
    entreprises = [
        {'ticker': 'AAPL', 'name': 'Apple'},
        {'ticker': 'MSFT', 'name': 'Microsoft'},
        {'ticker': 'TSLA', 'name': 'Tesla'},
        {'ticker': 'NVDA', 'name': 'Nvidia'},
    ]

    # Analyse de chaque entreprise
    all_dataframes = []
    
    for entreprise in entreprises:
        print(f"\nAnalyse de {entreprise['name']} ({entreprise['ticker']})...")
        df = analyzer.analyze_company(entreprise['ticker'], entreprise['name'])
        
        if df is not None and not df.empty:
            all_dataframes.append(df)
            
            # Afficher les meilleurs articles
            print(f"\nTop 3 articles:")
            top_articles = df.nlargest(3, 'score_pondere')
            for idx, (_, article) in enumerate(top_articles.iterrows(), 1):
                print(f"\n{idx}. {article['titre']}")
                print(f"   Score: {article['score']:.3f} | Source: {article['source']}")
                print(f"   Date: {article['date']}")
        
        print("\n" + "=" * 60)

    # Sauvegarder un fichier combine si on a des donnees
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_filename = f"{Config.SENTIMENT_FOLDER}/ALL_ARTICLES_{datetime.now().strftime('%Y%m%d')}.csv"
        combined_df.to_csv(combined_filename, index=False)
        print(f"Fichier combine sauvegarde: {combined_filename}")

    print("\nANALYSE TERMINEE")
    
    # Push vers GitHub
    if Config.PUSH_TOKEN:
        print("\n" + "="*60)
        print("PUSH VERS GITHUB...")
        success = git_push_to_repo()
        if success:
            print("SUCCES: Articles pousses vers GitHub")
        else:
            print("ECHEC: Probleme avec le push GitHub")
        print("="*60)
    
    return all_dataframes

# ==================== LANCEMENT ====================
if __name__ == "__main__":
    # Test connexion
    print("Test de connexion NewsAPI...")
    
    try:
        test_url = f"https://newsapi.org/v2/everything?q=test&apiKey={Config.NEWSAPI_KEY}&pageSize=1"
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            print("Connexion NewsAPI OK")
        else:
            print(f"Erreur NewsAPI: {response.status_code}")
            print(f"Message: {response.text[:200]}")
    except Exception as e:
        print(f"Connexion NewsAPI echouee: {e}")

    print("\n" + "=" * 60)  

    # Lancement
    main()

    print("\nUtilisation des resultats:")
    print("1. Les fichiers CSV contiennent tous les articles")
    print("2. Colonnes: date, source, titre, contenu, score, sentiment")
    print("3. Donnees sauvegardees dans: sentiment_data/")
    print("4. Poussees automatiquement sur GitHub")
