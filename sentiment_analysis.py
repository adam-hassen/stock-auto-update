# -*- coding: utf-8 -*-
"""
Analyse de sentiment simplifiee
Stockage dans data/articles_sentiment/
Avec colonne symbol et gestion cumulative
"""

import requests
from transformers import pipeline
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
import time
import random
import json
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

# ==================== FONCTIONS UTILITAIRES ====================
def load_existing_articles(ticker):
    """Charge les articles existants pour un ticker donne"""
    try:
        # Chercher le dernier fichier pour ce ticker
        files = [f for f in os.listdir(Config.ARTICLES_FOLDER) 
                if f.startswith(f'articles_{ticker}_') and f.endswith('.csv')]
        
        if not files:
            return pd.DataFrame()
        
        # Prendre le fichier le plus recent
        latest_file = sorted(files)[-1]
        file_path = os.path.join(Config.ARTICLES_FOLDER, latest_file)
        
        df = pd.read_csv(file_path)
        # Marquer comme anciens articles
        df['nouveau'] = False
        return df
        
    except Exception as e:
        print(f"Erreur chargement articles existants pour {ticker}: {e}")
        return pd.DataFrame()

def save_combined_articles(all_dataframes):
    """Sauvegarde combinee avec gestion cumulative"""
    if not all_dataframes:
        return
    
    # Nom du fichier combine
    combined_filename = f"{Config.ARTICLES_FOLDER}/ALL_ARTICLES_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    
    # Concatener tous les DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Sauvegarde du fichier du jour
    combined_df.to_csv(combined_filename, index=False)
    print(f"Fichier combine du jour sauvegarde: {combined_filename}")
    
    # Gestion du fichier master cumulatif
    master_filename = f"{Config.ARTICLES_FOLDER}/ALL_ARTICLES_MASTER.csv"
    
    try:
        # Charger le fichier master existant s'il existe
        if os.path.exists(master_filename):
            master_df = pd.read_csv(master_filename)
            print(f"Chargement du fichier master avec {len(master_df)} articles existants")
        else:
            master_df = pd.DataFrame()
            print("Creation d'un nouveau fichier master")
        
        # Ajouter la colonne 'ajout_date' pour les nouveaux articles
        today_date = datetime.now().strftime('%Y-%m-%d')
        combined_df['ajout_date'] = today_date
        
        # Fusionner avec le master
        if not master_df.empty:
            # Eviter les doublons bases sur titre + source + date
            mask = ~combined_df['titre'].isin(master_df['titre'])
            new_articles = combined_df[mask].copy()
            print(f"  {len(new_articles)} nouveaux articles a ajouter au master")
            
            if not new_articles.empty:
                # Ajouter les nouveaux articles
                updated_master = pd.concat([master_df, new_articles], ignore_index=True)
                updated_master.to_csv(master_filename, index=False)
                print(f"  Master mis a jour: {len(updated_master)} articles au total")
            else:
                print("  Aucun nouvel article a ajouter au master")
                updated_master = master_df
        else:
            # Premier enregistrement
            combined_df.to_csv(master_filename, index=False)
            updated_master = combined_df
            print(f"  Master cree avec {len(combined_df)} articles")
        
        # Statistiques du master
        print(f"STATISTIQUES MASTER:")
        print(f"  Total articles: {len(updated_master)}")
        print(f"  Entreprises uniques: {updated_master['symbol'].nunique()}")
        
        # Statistiques par date d'ajout
        if 'ajout_date' in updated_master.columns:
            date_stats = updated_master['ajout_date'].value_counts().sort_index()
            print(f"  Articles par date d'ajout:")
            for date, count in date_stats.items():
                print(f"    {date}: {count} articles")
        
    except Exception as e:
        print(f"Erreur gestion fichier master: {e}")
        import traceback
        traceback.print_exc()

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
            return None
        
        # Charger les anciens articles pour eviter les doublons
        old_articles = load_existing_articles(ticker)
        
        analyzed_articles = []
        new_articles_count = 0
        
        print(f"{len(articles)} articles trouves:")
        
        for i, article in enumerate(articles, 1):
            title = str(article.get('title', 'Sans titre')).strip()
            description = str(article.get('description', '')).strip()
            content = str(article.get('content', '')).strip()
            
            if content:
                full_text = f"{title}. {content}"
            else:
                full_text = f"{title}. {description}"
            
            # Verifier si l'article existe deja
            is_new = True
            if not old_articles.empty:
                # Verifier par titre (simplifie)
                if title in old_articles['titre'].values:
                    is_new = False
            
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
            
            source_name = article.get('source', {}).get('name', 'Inconnue')
            api_source = article.get('api_source', 'unknown')
            
            analyzed_article = {
                'symbol': ticker,  # AJOUT DE LA COLONNE SYMBOL
                'company': company_name,
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
                'url': article.get('url', ''),
                'nouveau': is_new,  # Marquer si c'est un nouvel article
                'analyse_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            analyzed_articles.append(analyzed_article)
            
            if is_new:
                new_articles_count += 1
                print(f"Article {i} [NOUVEAU]: {title[:50]}...")
            else:
                print(f"Article {i} [DEJA VU]: {title[:50]}...")
            
            print(f"  Score: {sentiment['score']:.3f} | Sentiment: {sentiment['label']} | Source: {source_name} | API: {api_source}")
        
        df = pd.DataFrame(analyzed_articles)
        
        if not df.empty:
            # Sauvegarde dans le dossier data/articles_sentiment
            filename = f"{Config.ARTICLES_FOLDER}/articles_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Donnees sauvegardees: {filename}")
            
            # Statistiques
            print(f"STATISTIQUES {ticker}:")
            print(f"  Total articles: {len(df)}")
            print(f"  Nouveaux articles: {new_articles_count}")
            print(f"  Articles deja existants: {len(df) - new_articles_count}")
            
            if 'api_source' in df.columns:
                print(f"  Repartition par source:")
                source_stats = df['api_source'].value_counts()
                for source, count in source_stats.items():
                    print(f"    {source}: {count} articles")
            
            positif = len(df[df['score'] > 0.1])
            negatif = len(df[df['score'] < -0.1])
            neutre = len(df) - positif - negatif
            
            print(f"  Resume sentiment:")
            print(f"    Articles positifs: {positif}")
            print(f"    Articles negatifs: {negatif}")
            print(f"    Articles neutres: {neutre}")
            print(f"    Score moyen: {df['score'].mean():.3f}")
        
        return df

# ==================== EXECUTION PRINCIPALE ====================
def main():
    print("DEBUT DE L'ANALYSE DE SENTIMENT")
    print("="*60)
    
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Periode: {Config.DAYS_BACK} derniers jours")
    print(f"Sources: NewsAPI={Config.USE_NEWSAPI}, GNews={Config.USE_GNEWS}")
    print(f"Articles par source: {Config.MAX_ARTICLES_PER_SOURCE}")
    print(f"Dossier de sauvegarde: {Config.ARTICLES_FOLDER}")
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
    
    all_dataframes = []
    
    for entreprise in entreprises:
        print(f"\n" + "="*60)
        print(f"ANALYSE DE {entreprise['name']} ({entreprise['ticker']})")
        print("="*60)
        
        df = analyzer.analyze_company(entreprise['ticker'], entreprise['name'])
        
        if df is not None and not df.empty:
            all_dataframes.append(df)
    
    if all_dataframes:
        # Sauvegarde combinee avec gestion cumulative
        save_combined_articles(all_dataframes)
        
        # Resume final
        print(f"\n" + "="*60)
        print("RESUME FINAL DE L'ANALYSE")
        print("="*60)
        
        total_articles = sum(len(df) for df in all_dataframes)
        new_articles = sum(df['nouveau'].sum() for df in all_dataframes if 'nouveau' in df.columns)
        
        print(f"Total articles analyses aujourd'hui: {total_articles}")
        print(f"Nouveaux articles: {new_articles}")
        print(f"Articles deja existants: {total_articles - new_articles}")
        print(f"Entreprises analysees: {len(entreprises)}")
        
        # Verifier si le fichier master existe
        master_filename = f"{Config.ARTICLES_FOLDER}/ALL_ARTICLES_MASTER.csv"
        if os.path.exists(master_filename):
            master_df = pd.read_csv(master_filename)
            print(f"\nFICHIER MASTER CUMULATIF:")
            print(f"  Total articles: {len(master_df)}")
            print(f"  Entreprises uniques: {master_df['symbol'].nunique()}")
    
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
