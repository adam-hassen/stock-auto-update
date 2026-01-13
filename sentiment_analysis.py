# -*- coding: utf-8 -*-
"""
Analyse de sentiment simplifiee
Stockage cumulatif dans un seul fichier master
"""

import requests
from transformers import pipeline
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
import time
import random
import hashlib
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    """Configuration centrale du projet"""
    
    USE_NEWSAPI = True
    USE_GNEWS = True
    
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
    GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
    
    MAX_ARTICLES_PER_SOURCE = 8
    DAYS_BACK = 3
    FINBERT_MODEL = "ProsusAI/finbert"
    
    # Dossier pour les articles
    ARTICLES_FOLDER = "data/articles_sentiment"
    os.makedirs(ARTICLES_FOLDER, exist_ok=True)
    
    # Fichier master unique
    MASTER_FILE = f"{ARTICLES_FOLDER}/ARTICLES_MASTER.csv"

# ==================== FONCTIONS UTILITAIRES ====================
def generate_article_id(title, source, date):
    """Genere un ID unique pour un article"""
    # Creer une chaine unique a partir du titre, source et date
    unique_string = f"{title}_{source}_{date}"
    # Generer un hash MD5
    return hashlib.md5(unique_string.encode()).hexdigest()[:16]

def load_master_articles():
    """Charge le fichier master existant"""
    try:
        if os.path.exists(Config.MASTER_FILE):
            df = pd.read_csv(Config.MASTER_FILE)
            print(f"Chargement du fichier master avec {len(df)} articles existants")
            return df
        else:
            print("Creation d'un nouveau fichier master")
            return pd.DataFrame()
    except Exception as e:
        print(f"Erreur chargement fichier master: {e}")
        return pd.DataFrame()

def save_to_master(new_articles_df):
    """Ajoute les nouveaux articles au fichier master"""
    try:
        # Charger les articles existants
        master_df = load_master_articles()
        
        if master_df.empty:
            # Premier enregistrement
            master_df = new_articles_df
        else:
            # Filtrer les articles qui n'existent pas deja
            # On compare par ID unique
            existing_ids = set(master_df['article_id'].values) if 'article_id' in master_df.columns else set()
            new_articles_df = new_articles_df[~new_articles_df['article_id'].isin(existing_ids)]
            
            if not new_articles_df.empty:
                # Ajouter les nouveaux articles
                master_df = pd.concat([master_df, new_articles_df], ignore_index=True)
                print(f"Ajout de {len(new_articles_df)} nouveaux articles au master")
            else:
                print("Aucun nouvel article a ajouter")
        
        # Sauvegarder le fichier master
        master_df.to_csv(Config.MASTER_FILE, index=False)
        print(f"Fichier master sauvegarde: {Config.MASTER_FILE}")
        print(f"Total articles dans master: {len(master_df)}")
        
        return master_df, len(new_articles_df) if 'new_articles_df' in locals() else 0
        
    except Exception as e:
        print(f"Erreur sauvegarde fichier master: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), 0

def create_summary_file(master_df, new_articles_count):
    """Cree un fichier de resume pour aujourd'hui"""
    try:
        # Date d'aujourd'hui
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Filtrer les articles ajoutes aujourd'hui
        if 'ajout_date' in master_df.columns:
            today_articles = master_df[master_df['ajout_date'] == today]
        else:
            today_articles = master_df
        
        if not today_articles.empty:
            # Creer un fichier de resume pour aujourd'hui
            summary_filename = f"{Config.ARTICLES_FOLDER}/RESUME_{today}.csv"
            today_articles.to_csv(summary_filename, index=False)
            print(f"Resume du jour sauvegarde: {summary_filename}")
            
            # Creer un fichier JSON avec des statistiques
            stats = {
                'date': today,
                'total_articles': len(master_df),
                'new_articles_today': new_articles_count,
                'companies_analyzed': today_articles['symbol'].nunique(),
                'sentiment_distribution': {
                    'positive': len(today_articles[today_articles['sentiment'] == 'positive']),
                    'negative': len(today_articles[today_articles['sentiment'] == 'negative']),
                    'neutral': len(today_articles[today_articles['sentiment'] == 'neutral'])
                },
                'source_distribution': today_articles['api_source'].value_counts().to_dict()
            }
            
            stats_filename = f"{Config.ARTICLES_FOLDER}/STATS_{today}.json"
            import json
            with open(stats_filename, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Statistiques sauvegardees: {stats_filename}")
            
    except Exception as e:
        print(f"Erreur creation fichiers de resume: {e}")

# ==================== SOURCES D'ARTICLES ====================
class NewsAPIClient:
    """Client pour recuperer les articles via NewsAPI"""
    
    @staticmethod
    def get_articles(query, max_results=10):
        if not Config.USE_NEWSAPI or not Config.NEWSAPI_KEY:
            print("NewsAPI desactive ou cle manquante")
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
            print(f"Recherche NewsAPI: {query}")
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                print(f"NewsAPI: {len(articles)} articles trouves")
                for article in articles:
                    article['api_source'] = 'newsapi'
                return articles
            else:
                print(f"NewsAPI Erreur {response.status_code}")
                return []
        except Exception as e:
            print(f"Erreur connexion NewsAPI: {e}")
            return []

class GNewsClient:
    """Client pour recuperer les articles via GNews"""
    
    @staticmethod
    def get_articles(query, max_results=10):
        if not Config.USE_GNEWS or not Config.GNEWS_API_KEY:
            print("GNews desactive ou cle manquante")
            return []
        
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
            print(f"Recherche GNews: {query}")
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                print(f"GNews: {len(articles)} articles trouves")
                
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
                print(f"GNews Erreur {response.status_code}")
                return []
        except Exception as e:
            print(f"Erreur connexion GNews: {e}")
            return []

class ArticleFetcher:
    """Combine les articles de toutes les sources"""
    
    @staticmethod
    def get_all_articles(query, max_results_per_source=8):
        all_articles = []
        
        if Config.USE_NEWSAPI:
            newsapi_articles = NewsAPIClient.get_articles(query, max_results_per_source)
            all_articles.extend(newsapi_articles)
            time.sleep(1)
        
        if Config.USE_GNEWS:
            gnews_articles = GNewsClient.get_articles(query, max_results_per_source)
            all_articles.extend(gnews_articles)
            time.sleep(1)
        
        random.shuffle(all_articles)
        
        max_total = max_results_per_source * 2
        if len(all_articles) > max_total:
            all_articles = all_articles[:max_total]
        
        print(f"Total articles: {len(all_articles)} (NewsAPI: {Config.USE_NEWSAPI}, GNews: {Config.USE_GNEWS})")
        return all_articles

# ==================== ANALYSE SENTIMENT ====================
class SentimentAnalyzer:
    """Analyseur de sentiment avec FinBERT"""
    
    def __init__(self):
        print("CHARGEMENT DU MODELE FinBERT")
        
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=Config.FINBERT_MODEL,
                tokenizer=Config.FINBERT_MODEL,
                device=-1
            )
            print("Modele FinBERT charge avec succes!")
        except Exception as e:
            print(f"Erreur chargement FinBERT: {e}")
            self.sentiment_pipeline = None
    
    def build_query(self, ticker, company_name):
        return f'"{company_name}" OR {ticker}'
    
    def analyze_text(self, text):
        try:
            text = str(text).strip()
            if len(text) < 20:
                return {'score': 0, 'label': 'neutral', 'confidence': 0.1}
            
            if len(text) > 500:
                text = text[:400] + " ... " + text[-100:]
            
            if self.sentiment_pipeline:
                result = self.sentiment_pipeline(text)[0]
                label = result['label']
                confidence = result['score']
            else:
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
            
            if label == 'positive':
                score = confidence if 'confidence' in locals() else 0.7
            elif label == 'negative':
                score = -confidence if 'confidence' in locals() else -0.7
            else:
                score = 0
            
            score = max(-1.0, min(1.0, score))
            
            return {
                'score': score,
                'label': label,
                'confidence': confidence if 'confidence' in locals() else 0.5
            }
            
        except Exception as e:
            print(f"Erreur analyse: {e}")
            return {'score': 0, 'label': 'neutral', 'confidence': 0.1}
    
    def calculate_weight(self, article):
        weight = 1.0
        
        api_source = article.get('api_source', 'unknown')
        if api_source == 'newsapi':
            weight *= 1.2
        elif api_source == 'gnews':
            weight *= 1.0
        
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
        
        source_name = article.get('source', {}).get('name', '').lower()
        
        important_sources = ['reuters', 'bloomberg', 'cnbc', 'financial times', 'wall street journal', 'yahoo finance']
        if any(source in source_name for source in important_sources):
            weight *= 1.5
        elif any(blog in source_name for blog in ['blog', 'personal', 'medium']):
            weight *= 0.6
        
        title = str(article.get('title', ''))
        description = str(article.get('description', ''))
        content = str(article.get('content', ''))
        content_length = len(title + description + content)
        
        if content_length > 300:
            weight *= 1.2
        elif content_length < 100:
            weight *= 0.8
        
        return max(0.2, min(weight, 3.0))
    
    def analyze_company(self, ticker, company_name):
        print(f"ANALYSE POUR {company_name} ({ticker})")
        
        query = self.build_query(ticker, company_name)
        articles = ArticleFetcher.get_all_articles(query, Config.MAX_ARTICLES_PER_SOURCE)
        
        if not articles:
            print("Aucun article trouve")
            return pd.DataFrame()
        
        analyzed_articles = []
        
        print(f"{len(articles)} articles trouves:")
        
        for i, article in enumerate(articles, 1):
            title = str(article.get('title', 'Sans titre')).strip()
            description = str(article.get('description', '')).strip()
            content = str(article.get('content', '')).strip()
            
            if content:
                full_text = f"{title}. {content}"
            else:
                full_text = f"{title}. {description}"
            
            # Generer un ID unique pour l'article
            source_name = article.get('source', {}).get('name', 'Inconnue')
            article_id = generate_article_id(title, source_name, 
                                           article.get('publishedAt', ''))
            
            sentiment = self.analyze_text(full_text)
            weight = self.calculate_weight(article)
            weighted_score = sentiment['score'] * weight
            
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
            
            api_source = article.get('api_source', 'unknown')
            
            analyzed_article = {
                'article_id': article_id,  # ID unique
                'symbol': ticker,
                'company': company_name,
                'date_publication': pub_date,
                'source': str(source_name),
                'api_source': api_source,
                'titre': title[:150] + '...' if len(title) > 150 else title,
                'contenu': full_text[:300] + '...' if len(full_text) > 300 else full_text,
                'score_sentiment': round(sentiment['score'], 3),
                'sentiment': sentiment['label'],
                'confiance': round(sentiment['confidence'], 3),
                'poids': round(weight, 2),
                'score_pondere': round(weighted_score, 3),
                'url': article.get('url', ''),
                'date_analyse': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ajout_date': datetime.now().strftime('%Y-%m-%d')  # Pour colorer dans Excel
            }
            
            analyzed_articles.append(analyzed_article)
            
            print(f"Article {i}: {title[:60]}...")
            print(f"  ID: {article_id}")
            print(f"  Score: {sentiment['score']:.3f} | Sentiment: {sentiment['label']} | Source: {source_name}")
        
        df = pd.DataFrame(analyzed_articles)
        
        if not df.empty:
            print(f"STATISTIQUES {ticker}:")
            print(f"  Articles trouves: {len(df)}")
            
            positif = len(df[df['sentiment'] == 'positive'])
            negatif = len(df[df['sentiment'] == 'negative'])
            neutre = len(df) - positif - negatif
            
            print(f"  Sentiment: {positif} positif, {negatif} negatif, {neutre} neutre")
            print(f"  Score moyen: {df['score_sentiment'].mean():.3f}")
        
        return df

# ==================== EXECUTION PRINCIPALE ====================
def main():
    print("DEBUT DE L'ANALYSE DE SENTIMENT")
    print("="*60)
    
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Periode analyse: {Config.DAYS_BACK} derniers jours")
    print(f"Sources: NewsAPI={Config.USE_NEWSAPI}, GNews={Config.USE_GNEWS}")
    print(f"Articles max/source: {Config.MAX_ARTICLES_PER_SOURCE}")
    print(f"Fichier master: {Config.MASTER_FILE}")
    print("="*60)
    
    test_apis()
    
    analyzer = SentimentAnalyzer()
    
    entreprises = [
        {'ticker': 'AAPL', 'name': 'Apple'},
        {'ticker': 'MSFT', 'name': 'Microsoft'},
        {'ticker': 'TSLA', 'name': 'Tesla'},
        {'ticker': 'NVDA', 'name': 'Nvidia'},
        {'ticker': 'GOOGL', 'name': 'Google'},
        {'ticker': 'AMZN', 'name': 'Amazon'},
        {'ticker': 'META', 'name': 'Meta'},
    ]
    
    # Charger le master existant pour stats initiales
    master_before = load_master_articles()
    initial_count = len(master_before) if not master_before.empty else 0
    print(f"\nArticles existants dans master: {initial_count}")
    
    all_new_articles = []
    
    for entreprise in entreprises:
        print(f"\n" + "="*60)
        print(f"ANALYSE DE {entreprise['name']} ({entreprise['ticker']})")
        print("="*60)
        
        df = analyzer.analyze_company(entreprise['ticker'], entreprise['name'])
        
        if not df.empty:
            all_new_articles.append(df)
    
    if all_new_articles:
        # Combiner tous les nouveaux articles
        combined_new = pd.concat(all_new_articles, ignore_index=True)
        print(f"\n" + "="*60)
        print(f"RESUME DE LA COLLECTE")
        print("="*60)
        print(f"Total articles collectes aujourd'hui: {len(combined_new)}")
        
        # Ajouter au fichier master
        master_df, new_count = save_to_master(combined_new)
        
        if not master_df.empty:
            print(f"\nSTATISTIQUES MASTER:")
            print(f"  Total articles: {len(master_df)}")
            print(f"  Nouveaux ajoutes aujourd'hui: {new_count}")
            print(f"  Entreprises uniques: {master_df['symbol'].nunique()}")
            
            # Statistiques de sentiment
            print(f"\nDISTRIBUTION DES SENTIMENTS:")
            sentiment_counts = master_df['sentiment'].value_counts()
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(master_df)) * 100
                print(f"  {sentiment}: {count} articles ({percentage:.1f}%)")
            
            # Creer un fichier de resume pour aujourd'hui
            create_summary_file(master_df, new_count)
            
            # Astuce pour colorer dans Excel/Sheets
            print(f"\nASTUCE POUR COLORER DANS EXCEL/SHEETS:")
            print(f"  1. Ouvrir le fichier: {Config.MASTER_FILE}")
            print(f"  2. Trier par la colonne 'ajout_date' (decroissant)")
            print(f"  3. Mettre en couleur les lignes ou 'ajout_date' = {today}")
            print(f"  4. Les nouveaux articles seront en haut et colores!")
            
            # Generer un petit script pour Excel
            excel_tips = f"""Pour colorer les nouveaux articles dans Excel:
1. Ouvrir """"{Config.MASTER_FILE}
2. Selectionner """toute la feuille (Ctrl+A)
3. Aller dans "Accueil" > "Mise en forme conditionnelle"
4. Choisir "Nouvelle regle" > "Utiliser une formule..."
5. Entrer: =$Q2= {today}   (si Q est la colonne ajout_date)
6. Choisir une couleur (ex: vert clair)
7. OK

Les nouveaux articles d'aujourd'hui seront colores!"""
            
            tips_file = f"{Config.ARTICLES_FOLDER}/EXCEL_TIPS.txt"
            with open(tips_file, 'w') as f:
                f.write(excel_tips)
            print(f"\nAstuces Excel sauvegardees: {tips_file}")
    
    print("\n" + "="*60)
    print("ANALYSE TERMINEE")
    print("="*60)

def test_apis():
    print("TEST DES APIS...")
    
    if Config.USE_NEWSAPI and Config.NEWSAPI_KEY:
        print(f"Test NewsAPI (cle: {Config.NEWSAPI_KEY[:10]}...)")
        test_url = f"https://newsapi.org/v2/everything?q=apple&apiKey={Config.NEWSAPI_KEY}&pageSize=1"
        try:
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                print("NewsAPI: Connexion OK")
            else:
                print(f"NewsAPI: Erreur {response.status_code}")
        except Exception as e:
            print(f"NewsAPI: {e}")
    else:
        print("NewsAPI desactive")
    
    if Config.USE_GNEWS and Config.GNEWS_API_KEY:
        print(f"Test GNews (cle: {Config.GNEWS_API_KEY[:10]}...)")
        test_url = f"https://gnews.io/api/v4/top-headlines?token={Config.GNEWS_API_KEY}&lang=en&max=1"
        try:
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                print("GNews: Connexion OK")
            else:
                print(f"GNews: Erreur {response.status_code}")
        except Exception as e:
            print(f"GNews: {e}")
    else:
        print("GNews desactive")

# ==================== LANCEMENT ====================
if __name__ == "__main__":
    print("CONFIGURATION DU SCRIPT")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"NewsAPI: {'Active' if Config.USE_NEWSAPI else 'Desactive'}")
    print(f"GNews: {'Active' if Config.USE_GNEWS else 'Desactive'}")
    print(f"Articles/source: {Config.MAX_ARTICLES_PER_SOURCE}")
    print(f"Periode: {Config.DAYS_BACK} jours")
    print(f"Modele: {Config.FINBERT_MODEL}")
    print("="*60)
    
    main()
