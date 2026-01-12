# ==================== IMPORT DES LIBRAIRIES ====================
import requests
from transformers import pipeline
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
import subprocess
import tempfile
import shutil
import glob
import time
import random
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    """Configuration centrale du projet"""
    
    # Sources d'articles (on utilise les deux)
    USE_NEWSAPI = True
    USE_GNEWS = True
    
    # Clés API
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', "ed172154b55e4d4eb95db4ac7895b29e")  # Ta nouvelle clé
    GNEWS_API_KEY = os.getenv('GNEWS_API_KEY', "7fd1230f54a4be2030c483514e145263")  # Clé GNews par défaut
    
    # Paramètres
    MAX_ARTICLES_PER_SOURCE = 8  # Max par source
    DAYS_BACK = 3
    FINBERT_MODEL = "ProsusAI/finbert"
    
    # Configuration GitHub
    PUSH_TOKEN = os.getenv('PUSH_TOKEN')

# ==================== FONCTIONS GIT AUTOMATISATION ====================
def push_articles_to_github():
    """Pousse automatiquement les fichiers CSV vers les deux repos GitHub"""
    
    if not Config.PUSH_TOKEN:
        print("\n" + "="*60)
        print("⚠️  PUSH_TOKEN non configuré")
        print("Les articles ne seront pas poussés vers GitHub")
        print("="*60)
        return False, False
    
    print("\n" + "="*60)
    print("🚀 DÉBUT AUTOMATISATION GITHUB")
    print("="*60)
    
    # Chercher tous les fichiers CSV d'articles générés
    csv_files = glob.glob("articles_*.csv")
    
    if not csv_files:
        print("❌ Aucun fichier CSV trouvé à pousser")
        return False, False
    
    print(f"📁 Fichiers à pousser: {len(csv_files)}")
    for csv_file in csv_files:
        print(f"  • {os.path.basename(csv_file)}")
    
    # ==================== REPO 1: adam-hassen/stock-auto-update (main) ====================
    print("\n" + "="*40)
    print("📦 REPO 1: adam-hassen/stock-auto-update")
    print("="*40)
    
    success_repo1 = push_to_repo(
        csv_files=csv_files,
        repo_owner="adam-hassen",
        repo_name="stock-auto-update",
        repo_branch="main",
        folder_name="sentiment_articles",
        bot_name="Sentiment Analysis Bot"
    )
    
    # ==================== REPO 2: Gasthorn/Projet4A_PredictionsBoursieres (Collecte-Des-Données) ====================
    print("\n" + "="*40)
    print("📦 REPO 2: Gasthorn/Projet4A_PredictionsBoursieres")
    print("="*40)
    
    success_repo2 = push_to_repo(
        csv_files=csv_files,
        repo_owner="Gasthorn",
        repo_name="Projet4A_PredictionsBoursieres",
        repo_branch="Collecte-Des-Donnees",
        folder_name="sentiment_articles",
        bot_name="Sentiment Analysis Bot"
    )
    
    return success_repo1, success_repo2

def push_to_repo(csv_files, repo_owner, repo_name, repo_branch, folder_name, bot_name):
    """Pousse les fichiers vers un repo spécifique"""
    
    print(f"  📍 Repo: {repo_owner}/{repo_name}")
    print(f"  🌿 Branche: {repo_branch}")
    print(f"  📂 Dossier: {folder_name}/")
    
    temp_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    
    try:
        print(f"  📁 Dossier temporaire: {temp_dir}")
        os.chdir(temp_dir)
        
        # Initialiser un nouveau repo git
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "sentiment-bot@github.com"], check=False)
        subprocess.run(["git", "config", "user.name", bot_name], check=False)
        
        # Créer la structure de dossiers
        os.makedirs(folder_name, exist_ok=True)
        
        # Copier chaque fichier CSV
        print("  📤 Copie des fichiers...")
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            dest_path = os.path.join(folder_name, filename)
            shutil.copy2(csv_file, dest_path)
            print(f"    → {folder_name}/{filename}")
        
        # Créer un README simple
        readme_content = f"""# 📰 Articles d'Actualités Financières

*Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## 📊 Description

Ce dossier contient les analyses de sentiment des articles financiers.
Les articles sont récupérés depuis **NewsAPI et GNews** et analysés avec le modèle **FinBERT**.

## 🏢 Entreprises analysées

- Apple (AAPL)
- Microsoft (MSFT)
- Tesla (TSLA)
- Nvidia (NVDA)

## 📋 Fichiers disponibles

Chaque fichier CSV contient les articles analysés pour une entreprise.
Chaque article inclut:
- Date de publication
- Source (Reuters, Bloomberg, etc.)
- Titre et contenu
- Score de sentiment (-1 à +1)
- Sentiment (positive/negative/neutral)
- Niveau de confiance
- Source de l'article (NewsAPI ou GNews)

## 🔧 Sources des données

- **NewsAPI**: {Config.NEWSAPI_KEY[:10]}...
- **GNews**: {Config.GNEWS_API_KEY[:10]}...
- Période: {Config.DAYS_BACK} derniers jours
- Analyse: **FinBERT** (ProsusAI/finbert)

---
*🤖 Généré automatiquement par GitHub Actions*
"""

        with open(f"{folder_name}/README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # Ajouter un .gitignore
        with open(".gitignore", "w") as f:
            f.write("*.pyc\n__pycache__/\n*.log\n.DS_Store\n")
        
        # Ajouter tous les fichiers
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        
        # Commit
        commit_msg = f"📰 Mise à jour articles {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        commit_result = subprocess.run(["git", "commit", "-m", commit_msg], 
                                     capture_output=True, text=True)
        
        if commit_result.returncode != 0 and "nothing to commit" not in commit_result.stdout:
            print(f"  ❌ Erreur commit")
            return False
        
        # Créer la branche spécifiée
        subprocess.run(["git", "branch", "-M", repo_branch], check=True, capture_output=True)
        
        # URL du repo avec token
        repo_url = f"https://x-access-token:{Config.PUSH_TOKEN}@github.com/{repo_owner}/{repo_name}.git"
        
        # Ajouter le remote
        remote_add = subprocess.run(["git", "remote", "add", "origin", repo_url], 
                                  capture_output=True, text=True)
        
        if remote_add.returncode != 0:
            print("  ℹ️  Remote déjà configuré")
        
        # Force push vers la branche
        print(f"  🚀 Pushing vers {repo_branch}...")
        push_result = subprocess.run(["git", "push", "--force", "origin", repo_branch], 
                                   capture_output=True, text=True)
        
        if push_result.returncode == 0:
            print(f"  ✅ Push réussi!")
            print(f"  🔗 URL: https://github.com/{repo_owner}/{repo_name}/tree/{repo_branch}/{folder_name}")
            return True
        else:
            print(f"  ❌ Erreur push")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {str(e)[:100]}")
        return False
    finally:
        os.chdir(original_dir)
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

# ==================== SOURCES D'ARTICLES ====================
class NewsAPIClient:
    """Client pour récupérer les articles via NewsAPI"""
    
    @staticmethod
    def get_articles(query, max_results=10):
        """Récupère les articles via NewsAPI"""
        if not Config.USE_NEWSAPI or not Config.NEWSAPI_KEY:
            print("  ⚠️  NewsAPI désactivé ou clé manquante")
            return []
        
        from_date = (datetime.now() - timedelta(days=Config.DAYS_BACK)).strftime('%Y-%m-%d')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f'{query} AND (stock OR shares OR earnings OR revenue OR market)',
            'apiKey': Config.NEWSAPI_KEY,
            'pageSize': max_results,
            'language': 'en',
            'sortBy': 'relevancy',
            'from': from_date,
        }
        
        try:
            print(f"  🔍 Recherche NewsAPI: {query}")
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                print(f"  ✅ NewsAPI: {len(articles)} articles trouvés")
                # Ajouter la source
                for article in articles:
                    article['api_source'] = 'newsapi'
                return articles
            else:
                print(f"  ❌ NewsAPI Erreur {response.status_code}")
                if response.status_code == 401:
                    print(f"    Message: Clé API invalide")
                return []
        except Exception as e:
            print(f"  ❌ Erreur connexion NewsAPI: {e}")
            return []

class GNewsClient:
    """Client pour récupérer les articles via GNews"""
    
    @staticmethod
    def get_articles(query, max_results=10):
        """Récupère les articles via GNews"""
        if not Config.USE_GNEWS or not Config.GNEWS_API_KEY:
            print("  ⚠️  GNews désactivé ou clé manquante")
            return []
        
        # GNews API endpoint
        url = "https://gnews.io/api/v4/search"
        
        params = {
            'q': f'{query} stock market',
            'token': Config.GNEWS_API_KEY,
            'lang': 'en',
            'max': max_results,
            'from': (datetime.now() - timedelta(days=Config.DAYS_BACK)).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'to': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'sortby': 'relevance'
        }
        
        try:
            print(f"  🔍 Recherche GNews: {query}")
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                print(f"  ✅ GNews: {len(articles)} articles trouvés")
                
                # Formater les articles au même format que NewsAPI
                formatted_articles = []
                for article in articles:
                    formatted_article = {
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'publishedAt': article.get('publishedAt', ''),
                        'source': {'name': article.get('source', {}).get('name', 'GNews')},
                        'url': article.get('url', ''),
                        'api_source': 'gnews'
                    }
                    formatted_articles.append(formatted_article)
                
                return formatted_articles
            else:
                print(f"  ❌ GNews Erreur {response.status_code}")
                if 'errors' in response.json():
                    print(f"    Message: {response.json()['errors']}")
                return []
        except Exception as e:
            print(f"  ❌ Erreur connexion GNews: {e}")
            return []

class ArticleFetcher:
    """Combine les articles de toutes les sources"""
    
    @staticmethod
    def get_all_articles(query, max_results_per_source=8):
        """Récupère les articles de toutes les sources disponibles"""
        all_articles = []
        
        # Récupérer depuis NewsAPI
        if Config.USE_NEWSAPI:
            newsapi_articles = NewsAPIClient.get_articles(query, max_results_per_source)
            all_articles.extend(newsapi_articles)
            time.sleep(1)  # Pause pour éviter le rate limiting
        
        # Récupérer depuis GNews
        if Config.USE_GNEWS:
            gnews_articles = GNewsClient.get_articles(query, max_results_per_source)
            all_articles.extend(gnews_articles)
            time.sleep(1)
        
        # Mélanger les articles pour varier les sources
        random.shuffle(all_articles)
        
        # Limiter le nombre total d'articles
        max_total = max_results_per_source * 2
        if len(all_articles) > max_total:
            all_articles = all_articles[:max_total]
        
        print(f"  📊 Total articles: {len(all_articles)} (NewsAPI: {Config.USE_NEWSAPI}, GNews: {Config.USE_GNEWS})")
        return all_articles

# ==================== ANALYSE SENTIMENT ====================
class SentimentAnalyzer:
    """Analyseur de sentiment avec FinBERT"""
    
    def __init__(self):
        print("\n" + "="*60)
        print("🧠 CHARGEMENT DU MODÈLE FinBERT")
        print("="*60)
        
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=Config.FINBERT_MODEL,
                tokenizer=Config.FINBERT_MODEL,
                device=-1  # Force CPU pour éviter les problèmes
            )
            print("✅ Modèle FinBERT chargé avec succès!")
        except Exception as e:
            print(f"❌ Erreur chargement FinBERT: {e}")
            print("⚠️  Utilisation d'une analyse simple")
            self.sentiment_pipeline = None
    
    def build_query(self, ticker, company_name):
        """Construit une requête pour l'entreprise"""
        return f'"{company_name}" OR {ticker}'
    
    def analyze_text(self, text):
        """Analyse le sentiment d'un texte"""
        try:
            text = str(text).strip()
            if len(text) < 20:
                return {'score': 0, 'label': 'neutral', 'confidence': 0.1}
            
            # Limite la longueur du texte
            if len(text) > 500:
                text = text[:400] + " ... " + text[-100:]
            
            # Analyse avec FinBERT si disponible
            if self.sentiment_pipeline:
                result = self.sentiment_pipeline(text)[0]
                label = result['label']
                confidence = result['score']
            else:
                # Fallback simple si FinBERT échoue
                positive_words = ['profit', 'gain', 'growth', 'increase', 'rise', 'positive', 'strong', 'bullish']
                negative_words = ['loss', 'decline', 'decrease', 'fall', 'negative', 'weak', 'bearish']
                
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                total = positive_count + negative_count
                if total > 0:
                    score = (positive_count - negative_count) / total
                    if score > 0.1:
                        label = 'positive'
                    elif score < -0.1:
                        label = 'negative'
                    else:
                        label = 'neutral'
                    confidence = min(0.9, total / 10)
                else:
                    label = 'neutral'
                    confidence = 0.1
                    score = 0
            
            # Convertir en score numérique
            if label == 'positive':
                score = confidence if 'confidence' in locals() else 0.7
            elif label == 'negative':
                score = -confidence if 'confidence' in locals() else -0.7
            else:
                score = 0
            
            # Normaliser entre -1 et 1
            score = max(-1.0, min(1.0, score))
            
            return {
                'score': score,
                'label': label,
                'confidence': confidence if 'confidence' in locals() else 0.5
            }
            
        except Exception as e:
            print(f"  ⚠️  Erreur analyse: {e}")
            return {'score': 0, 'label': 'neutral', 'confidence': 0.1}
    
    def calculate_weight(self, article):
        """Calcule l'importance de l'article"""
        weight = 1.0
        
        # Source de l'API
        api_source = article.get('api_source', 'unknown')
        if api_source == 'newsapi':
            weight *= 1.2  # NewsAPI généralement plus fiable
        elif api_source == 'gnews':
            weight *= 1.0
        
        # 1. Actualité
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
        
        # 2. Source du média
        source_name = article.get('source', {}).get('name', '').lower()
        
        # Sources importantes
        important_sources = ['reuters', 'bloomberg', 'cnbc', 'financial times', 'wall street journal', 'yahoo finance']
        if any(source in source_name for source in important_sources):
            weight *= 1.5
        # Blogs personnels
        elif any(blog in source_name for blog in ['blog', 'personal', 'medium']):
            weight *= 0.6
        
        # 3. Longueur
        title = str(article.get('title', ''))
        description = str(article.get('description', ''))
        content = str(article.get('content', ''))
        content_length = len(title + description + content)
        
        if content_length > 300:
            weight *= 1.2
        elif content_length < 100:
            weight *= 0.8
        
        # Limites
        return max(0.2, min(weight, 3.0))
    
    def analyze_company(self, ticker, company_name):
        """Analyse les articles d'une entreprise"""
        print(f"\n{'='*60}")
        print(f"📊 ANALYSE POUR {company_name} ({ticker})")
        print(f"{'='*60}")
        
        # Récupération des articles depuis toutes les sources
        query = self.build_query(ticker, company_name)
        articles = ArticleFetcher.get_all_articles(query, Config.MAX_ARTICLES_PER_SOURCE)
        
        if not articles:
            print("❌ Aucun article trouvé")
            return None
        
        analyzed_articles = []
        
        print(f"\n📰 {len(articles)} articles trouvés:")
        print("-" * 60)
        
        for i, article in enumerate(articles, 1):
            # Texte complet
            title = str(article.get('title', 'Sans titre')).strip()
            description = str(article.get('description', '')).strip()
            content = str(article.get('content', '')).strip()
            
            # Utiliser le contenu le plus complet disponible
            if content:
                full_text = f"{title}. {content}"
            else:
                full_text = f"{title}. {description}"
            
            # Analyse sentiment
            sentiment = self.analyze_text(full_text)
            
            # Calcul poids
            weight = self.calculate_weight(article)
            
            # Score pondéré
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
            api_source = article.get('api_source', 'unknown')
            
            # Article complet
            analyzed_article = {
                'id': i,
                'date': pub_date,
                'source': str(source_name),
                'api_source': api_source,
                'titre': title[:100] + '...' if len(title) > 100 else title,
                'contenu': full_text[:200] + '...' if len(full_text) > 200 else full_text,
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
            print(f"  📊 Score: {sentiment['score']:.3f} | 🏷️  Sentiment: {sentiment['label']} | 📰 Source: {source_name} | 🔌 API: {api_source}")
        
        # Création du DataFrame
        df = pd.DataFrame(analyzed_articles)
        
        # Sauvegarde CSV
        if not df.empty:
            filename = f"articles_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"\n💾 Données sauvegardées: {filename}")
            
            # Statistiques par source API
            if 'api_source' in df.columns:
                print(f"\n📈 RÉPARTITION PAR SOURCE:")
                source_stats = df['api_source'].value_counts()
                for source, count in source_stats.items():
                    print(f"  • {source}: {count} articles")
            
            # Statistiques de sentiment
            positif = len(df[df['score'] > 0.1])
            negatif = len(df[df['score'] < -0.1])
            neutre = len(df) - positif - negatif
            
            print(f"\n📊 RÉSUMÉ SENTIMENT:")
            print(f"  ✅ Articles positifs: {positif}")
            print(f"  ❌ Articles négatifs: {negatif}")
            print(f"  ⚪ Articles neutres: {neutre}")
            print(f"  📈 Score moyen: {df['score'].mean():.3f}")
        
        return df

# ==================== EXECUTION PRINCIPALE ====================
def main():
    """Fonction principale"""
    print("\n" + "="*60)
    print("🚀 DÉBUT DE L'ANALYSE DE SENTIMENT")
    print("="*60)
    
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Période: {Config.DAYS_BACK} derniers jours")
    print(f"🔌 Sources activées: NewsAPI={Config.USE_NEWSAPI}, GNews={Config.USE_GNEWS}")
    print(f"📈 Articles par source: {Config.MAX_ARTICLES_PER_SOURCE}")
    print("="*60)
    
    # Test des connexions API
    print("\n🔍 TEST DES CONNEXIONS API...")
    test_apis()
    
    # Initialisation de l'analyseur
    analyzer = SentimentAnalyzer()
    
    # Entreprises à analyser
    entreprises = [
        {'ticker': 'AAPL', 'name': 'Apple'},
        {'ticker': 'MSFT', 'name': 'Microsoft'},
        {'ticker': 'TSLA', 'name': 'Tesla'},
        {'ticker': 'NVDA', 'name': 'Nvidia'},
        {'ticker': 'GOOGL', 'name': 'Google'},
        {'ticker': 'AMZN', 'name': 'Amazon'},
        {'ticker': 'META', 'name': 'Meta'},
    ]
    
    # Analyse de chaque entreprise
    all_dataframes = []
    
    for entreprise in entreprises:
        print(f"\n" + "="*60)
        print(f"🎯 ANALYSE DE {entreprise['name']} ({entreprise['ticker']})")
        print("="*60)
        
        df = analyzer.analyze_company(entreprise['ticker'], entreprise['name'])
        
        if df is not None and not df.empty:
            all_dataframes.append(df)
            
            # Afficher les meilleurs articles
            print(f"\n🏆 TOP 3 ARTICLES:")
            top_articles = df.nlargest(3, 'score_pondere')
            for idx, (_, article) in enumerate(top_articles.iterrows(), 1):
                print(f"\n{idx}. {article['titre']}")
                print(f"   📊 Score: {article['score']:.3f} | 🏷️  Sentiment: {article['sentiment']}")
                print(f"   📰 Source: {article['source']} | 🔌 API: {article.get('api_source', 'N/A')}")
                print(f"   📅 Date: {article['date']}")
    
    # Sauvegarde d'un fichier combiné
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_filename = f"ALL_ARTICLES_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        combined_df.to_csv(combined_filename, index=False)
        print(f"\n💾 Fichier combiné sauvegardé: {combined_filename}")
        
        # Statistiques globales
        print(f"\n" + "="*60)
        print("📊 STATISTIQUES GLOBALES")
        print("="*60)
        print(f"📈 Total articles analysés: {len(combined_df)}")
        print(f"🏢 Entreprises analysées: {len(entreprises)}")
        
        if 'api_source' in combined_df.columns:
            print(f"\n🔌 RÉPARTITION PAR SOURCE API:")
            source_stats = combined_df['api_source'].value_counts()
            for source, count in source_stats.items():
                percentage = (count / len(combined_df)) * 100
                print(f"  • {source}: {count} articles ({percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ ANALYSE TERMINÉE")
    print("="*60)
    
    # ==================== AUTOMATISATION GITHUB ====================
    # Pousse automatiquement les fichiers vers GitHub
    success_repo1, success_repo2 = push_articles_to_github()
    
    print("\n" + "="*60)
    print("🚀 RÉSUMÉ DE L'AUTOMATISATION GITHUB")
    print("="*60)
    
    if success_repo1:
        print("✅ REPO 1: adam-hassen/stock-auto-update")
        print("   🔗 https://github.com/adam-hassen/stock-auto-update/tree/main/sentiment_articles")
    else:
        print("❌ REPO 1: Échec")
    
    if success_repo2:
        print("✅ REPO 2: Gasthorn/Projet4A_PredictionsBoursieres")
        print("   🔗 https://github.com/Gasthorn/Projet4A_PredictionsBoursieres/tree/Collecte-Des-Donnees/sentiment_articles")
    else:
        print("❌ REPO 2: Échec")
    
    print("="*60)
    
    print("\n📋 UTILISATION DES RÉSULTATS:")
    print("1. 📁 Les fichiers CSV contiennent tous les articles")
    print("2. 📊 Colonnes: date, source, api_source, titre, contenu, score, sentiment")
    print("3. 🔌 Sources: NewsAPI et GNews combinées")
    print("4. 🚀 Poussés automatiquement sur DEUX repos GitHub")
    print("5. 📂 Dossier: sentiment_articles/")

def test_apis():
    """Teste les connexions aux différentes API"""
    
    print("\n🧪 TEST DES APIS...")
    
    # Test NewsAPI
    if Config.USE_NEWSAPI and Config.NEWSAPI_KEY:
        print(f"  🔍 Test NewsAPI (clé: {Config.NEWSAPI_KEY[:10]}...)")
        test_url = f"https://newsapi.org/v2/everything?q=apple&apiKey={Config.NEWSAPI_KEY}&pageSize=1"
        try:
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                print("  ✅ NewsAPI: Connexion OK")
            else:
                print(f"  ❌ NewsAPI: Erreur {response.status_code}")
                if response.status_code == 401:
                    print("     ⚠️  Clé API invalide ou expirée")
        except Exception as e:
            print(f"  ❌ NewsAPI: {e}")
    else:
        print("  ⚠️  NewsAPI désactivé")
    
    # Test GNews
    if Config.USE_GNEWS and Config.GNEWS_API_KEY:
        print(f"  🔍 Test GNews (clé: {Config.GNEWS_API_KEY[:10]}...)")
        test_url = f"https://gnews.io/api/v4/top-headlines?token={Config.GNEWS_API_KEY}&lang=en&max=1"
        try:
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                print("  ✅ GNews: Connexion OK")
            else:
                print(f"  ❌ GNews: Erreur {response.status_code}")
                if 'errors' in response.json():
                    print(f"     ⚠️  {response.json()['errors']}")
        except Exception as e:
            print(f"  ❌ GNews: {e}")
    else:
        print("  ⚠️  GNews désactivé")

# ==================== LANCEMENT ====================
if __name__ == "__main__":
    # Affichage de la configuration
    print("\n" + "="*60)
    print("⚙️  CONFIGURATION DU SCRIPT")
    print("="*60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔌 NewsAPI: {'✅ Activé' if Config.USE_NEWSAPI else '❌ Désactivé'}")
    print(f"🔌 GNews: {'✅ Activé' if Config.USE_GNEWS else '❌ Désactivé'}")
    print(f"📊 Articles/source: {Config.MAX_ARTICLES_PER_SOURCE}")
    print(f"📅 Période: {Config.DAYS_BACK} jours")
    print(f"🤖 Modèle: {Config.FINBERT_MODEL}")
    print(f"🔑 PUSH_TOKEN: {'✅ Présent' if Config.PUSH_TOKEN else '❌ Absent'}")
    print("="*60)
    
    # Lancement
    main()
