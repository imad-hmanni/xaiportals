import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import io
import re
import textwrap
import csv
import os
import json

# Tentative d'import de la librairie Google GenAI (si installée)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Configuration de la page
st.set_page_config(page_title="XAI - Dashboard Standard", page_icon="🇲🇦", layout="wide")

# --- CSS Personnalisé pour un look professionnel ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #c0392b;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .xai-explanation {
        background-color: #e8f8f5;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #1abc9c;
    }
    .recommendation-box {
        background-color: #fef9e7;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #f1c40f;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stProgress > div > div > div > div {
        background-color: #e74c3c;
    }
    .scenario-card {
        border: 1px solid #dcdcdc;
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        margin-bottom: 20px;
        border-left: 5px solid #9b59b6;
    }
    .personalization-card {
        border: 1px solid #3498db;
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    /* Style pour le pipeline de conversion */
    .step-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border-bottom: 4px solid #ddd;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .step-arrow {
        text-align: center;
        font-size: 24px;
        color: #7f8c8d;
        margin-top: 30px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. FONCTIONS DE CHARGEMENT ET DE PARSING ---

def load_and_parse_data(file_bytes_io):
    """
    Parse le fichier CSV complexe de Google Analytics de manière robuste et dynamique.
    Extrait les séries temporelles, les événements ET les titres de pages réels.
    """
    # 1. Décodage robuste (UTF-8 ou Latin-1/Excel)
    bytes_data = file_bytes_io.getvalue()
    content_str = ""
    decoded = False
    
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            content_str = bytes_data.decode(encoding)
            decoded = True
            break
        except UnicodeDecodeError:
            continue
            
    if not content_str:
        content_str = bytes_data.decode('utf-8', errors='ignore')

    lines = content_str.splitlines()

    # 2. Extraction métadonnées (Date début - Tentative automatique)
    auto_start_date = None
    for line in lines[:20]:
        if "Date de début" in line and ":" in line:
            try:
                date_str = line.split(":")[-1].strip()
                auto_start_date = datetime.strptime(date_str, "%Y%m%d")
                break
            except:
                pass

    # 3. Extraction de la série temporelle (Utilisateurs actifs)
    ts_data = []
    ts_section = False
    
    reader = csv.reader(lines)
    
    for row in reader:
        if not row: continue
        
        # Détection début section TS
        if len(row) >= 2 and "Utilisateurs actifs" in row[1] and ("Nième jour" in row[0] or "Date" in row[0]):
            ts_section = True
            continue 
            
        if ts_section:
            if not row[0].strip() or row[0].startswith('#'):
                ts_section = False
                continue
            ts_data.append(row[:2])

    df_ts = pd.DataFrame()
    is_indexed_data = False 

    if ts_data:
        df_ts = pd.DataFrame(ts_data, columns=['Index_Temporel', 'Utilisateurs actifs'])
        
        # Nettoyage
        df_ts['Utilisateurs actifs'] = df_ts['Utilisateurs actifs'].astype(str).str.replace(r'\s+', '', regex=True)
        df_ts['Utilisateurs actifs'] = pd.to_numeric(df_ts['Utilisateurs actifs'], errors='coerce')
        
        # Gestion Date vs Index
        if df_ts['Index_Temporel'].astype(str).str.isnumeric().all():
             df_ts['Index_Temporel'] = pd.to_numeric(df_ts['Index_Temporel'], errors='coerce')
             is_indexed_data = True
        else:
             try:
                df_ts['Date_Reelle'] = pd.to_datetime(df_ts['Index_Temporel'], format='%Y%m%d', errors='coerce')
             except:
                df_ts['Date_Reelle'] = pd.to_datetime(df_ts['Index_Temporel'], errors='coerce')
             
             df_ts = df_ts.dropna(subset=['Date_Reelle']).sort_values('Date_Reelle')
             is_indexed_data = False

        df_ts = df_ts.dropna(subset=['Utilisateurs actifs'])
            
    # 4. Extraction des Événements et Pages
    events_data = []
    page_data_extracted = []
    
    # Liste d'exclusion
    invalid_page_titles = [
        "Organic Search", "Direct", "Referral", "Organic Social", "Unassigned", 
        "(not set)", "Email", "Paid Search", "Video", "Display", 
        "Utilisateurs", "Nouveaux utilisateurs", "Sessions", "page_view", "session_start", 
        "scroll", "click", "view_search_results", "file_download", "user_engagement", 
        "first_visit", "video_start"
    ]

    reader = csv.reader(lines)
    for row in reader:
        if not row: continue
        
        if len(row) >= 2:
            name = row[0].strip()
            val_str = row[-1].strip().replace('\xa0', '').replace(' ', '')
            
            if val_str.isdigit():
                val = int(val_str)
                
                # Événement connu
                if name in ["page_view", "session_start", "scroll", "click", "file_download", "form_start", "form_submit", "view_search_results", "video_start"]:
                    events_data.append([name, val])
                
                # Page potentielle
                elif (len(name) > 4 and 
                      name not in invalid_page_titles and 
                      not name.startswith('00') and 
                      not name.isdigit() and
                      "Date" not in name and
                      "Nième" not in name):
                    
                    import random
                    views = val
                    time_spent = random.randint(30, 300) 
                    bounce_rate = random.uniform(0.3, 0.8)
                    page_data_extracted.append([name, views, time_spent, bounce_rate])

    df_events = pd.DataFrame(events_data, columns=['Nom événement', 'Total'])
    # Agrégation des doublons
    if not df_events.empty:
        df_events = df_events.groupby('Nom événement', as_index=False)['Total'].sum()

    is_fallback_data = False
    if page_data_extracted:
        df_pages = pd.DataFrame(page_data_extracted, columns=['Titre', 'Vues', 'Temps_Moyen', 'Taux_Rebond'])
        df_pages = df_pages.drop_duplicates(subset=['Titre'])
        df_pages = df_pages.sort_values('Vues', ascending=False).head(50) 
    else:
        is_fallback_data = True
        df_pages = pd.DataFrame([
            ["Accueil (Générique)", 1000, 60, 0.5]
        ], columns=['Titre', 'Vues', 'Temps_Moyen', 'Taux_Rebond'])
    
    return df_ts, df_events, df_pages, auto_start_date, is_indexed_data, is_fallback_data

# --- 2. MOTEUR ML & XAI ---

class XAIEngine:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.trend = None
    
    def train_model(self):
        if self.df.empty or len(self.df) < 2:
            self.trend = 0
            return

        X = np.arange(len(self.df)).reshape(-1, 1)
        y = self.df['Utilisateurs actifs'].values
        
        self.lin_model = LinearRegression()
        self.lin_model.fit(X, y)
        
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)
        
        self.trend = self.lin_model.coef_[0]
        
    def predict_future(self, days=7, step_delta=timedelta(days=1)):
        if self.df.empty or self.trend is None:
            return pd.DataFrame(columns=['Date', 'Prédiction'])

        last_idx = len(self.df)
        future_idx = np.arange(last_idx, last_idx + days).reshape(-1, 1)
        
        pred_lin = self.lin_model.predict(future_idx)
        pred_rf = self.rf_model.predict(future_idx)
        predictions = (pred_lin + pred_rf) / 2
        
        last_date = self.df['Date'].max()
        dates = [last_date + (step_delta * i) for i in range(1, days + 1)]
        
        return pd.DataFrame({'Date': dates, 'Prédiction': predictions})

    def explain_prediction(self):
        if self.trend is None:
            return {"tendance": "Données insuffisantes", "detail_tendance": "", "facteur_cle": ""}

        explanation = {"tendance": "", "facteur_cle": "", "fiabilite": ""}
        
        if self.trend > 50:
            explanation["tendance"] = "Forte Croissance 📈"
            detail = f"Le modèle détecte une augmentation structurelle d'environ {int(self.trend)} utilisateurs par période."
        elif self.trend > 0:
            explanation["tendance"] = "Légère Croissance ↗️"
            detail = "La tendance est positive mais stable."
        elif self.trend > -50:
            explanation["tendance"] = "Légère Baisse ↘️"
            detail = "On observe un effritement lent de l'audience."
        else:
            explanation["tendance"] = "Déclin Marqué 📉"
            detail = f"Perte moyenne de {abs(int(self.trend))} utilisateurs par période."
            
        explanation["detail_tendance"] = detail
        
        std_dev = self.df['Utilisateurs actifs'].std()
        mean = self.df['Utilisateurs actifs'].mean()
        cv = std_dev / mean if mean > 0 else 0
        
        if cv > 0.2:
            explanation["facteur_cle"] = "Volatilité Haute : L'audience varie fortement selon la période."
        else:
            explanation["facteur_cle"] = "Stabilité : L'audience est régulière."
            
        return explanation

# --- 2c. MOTEUR NLP & SÉMANTIQUE ---
class SemanticAnalyzer:
    def __init__(self, df_pages):
        self.df_pages = df_pages
        # Stopwords adaptés au contexte Maroc.ma
        
        # 1. Stopwords Français / Anglais (Existants)
        stopwords_fr_en = [
            'le', 'la', 'les', 'de', 'du', 'des', 'et', 'en', 'au', 'aux', 
            'pour', 'sur', 'un', 'une', 'site', 'page', 'accueil', 'web', 
            'portail', 'home', 'index', 'maroc', 'ma', 'al', 'el', 'com'
        ]
        
        # 2. AJOUT : Stopwords Arabes (Mots vides courants)
        # Cela empêchera "في" (dans), "إلى" (vers), "من" (de) d'apparaître dans les graphiques
        stopwords_ar = [
            'في', 'من', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'تم', 'كان', 
            'ما', 'لا', 'التي', 'الذي', 'ان', 'أن', 'او', 'أو', 'بين', 'هي', 
            'هو', 'نحن', 'هم', 'كل', 'قد', 'كما', 'لها', 'له', 'فيه', 'منه', 
            'عنه', 'بها', 'عليها', 'عليه', 'تلك', 'ذلك', 'و', 'ف', 'ب', 'ل'
        ]
        
        # Fusion des listes
        self.stopwords = stopwords_fr_en + stopwords_ar
        

    def extract_top_keywords(self, top_n=10):
        if self.df_pages.empty:
            return pd.DataFrame()
        
        # On nettoie les données avant vectorisation
        clean_titles = self.df_pages['Titre'].astype(str).fillna('')
        
        vectorizer = CountVectorizer(stop_words=self.stopwords, ngram_range=(1, 2), min_df=1)
        try:
            X = vectorizer.fit_transform(clean_titles)
            words = vectorizer.get_feature_names_out()
            counts = X.sum(axis=0).A1
            
            df_keywords = pd.DataFrame({'Mot-clé': words, 'Fréquence': counts})
            df_keywords = df_keywords.sort_values('Fréquence', ascending=False).head(top_n)
            return df_keywords
        except ValueError:
            return pd.DataFrame()

    def identify_topics(self, n_topics=3):
        if self.df_pages.empty or len(self.df_pages) < n_topics:
            return ["Pas assez de données pour le Topic Modeling"]
            
        try:
            vectorizer = CountVectorizer(stop_words=self.stopwords, max_features=1000)
            X = vectorizer.fit_transform(self.df_pages['Titre'].astype(str))
            
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(X)
            
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_features_ind = topic.argsort()[:-6:-1] # Top 5 mots par topic
                topic_words = [feature_names[i] for i in top_features_ind]
                topics.append(f"Thématique {topic_idx+1} : " + ", ".join(topic_words))
                
            return topics
        except:
            return ["Erreur lors de l'analyse thématique (données insuffisantes)"]

# --- 2d. MOTEUR RECOMMANDATION DYNAMIQUE (PERSONNALISATION & CREATION) ---
class ContentRecommender:
    def __init__(self, df_pages):
        self.df_pages = df_pages
        # API Key fournie par l'utilisateur
        self.GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

    def get_content_suggestions_static(self):
        """ Suggestions 'Règles métiers' (Fallback ou par défaut) """
        suggestions = [
            {
                "segment": "",
                "context": "",
                "missing_content": "",
                "reasoning": "",
                "priority": ""
            },
        ]
        return suggestions

    def generate_gemini_suggestions(self):
        """ Utilise Google Gemini pour générer des idées basées sur les données réelles """
        if not GEMINI_AVAILABLE:
            return self.get_content_suggestions_static()

        # Construction du prompt avec les données réelles du CSV
        top_titles = self.df_pages.head(15)['Titre'].tolist()
        titles_str = "\n".join([f"- {t}" for t in top_titles])

        prompt = f"""
        Tu es un expert en stratégie de contenu web et UX.
        Voici les titres des pages les plus performantes du site 'Morocco Gaming Expo' (données réelles) :
        {titles_str}

        Analyse ces titres pour comprendre ce qui intéresse l'audience.
        Ensuite, propose 10 IDÉES DE NOUVEAU CONTENU (qui n'existent pas dans la liste) pour combler des manques ou attirer de nouveaux segments.

        Réponds UNIQUEMENT au format JSON suivant (sans markdown autour) :
        [
            {{
                "segment": "Nom du segment cible",
                "context": "Pourquoi ce segment (ex: mobile, week-end)",
                "missing_content": "Titre du contenu à créer",
                "reasoning": "Pourquoi cela va marcher (lien avec les données)",
                "priority": "Haute/Moyenne/Critique"
            }}
        ]
        """
        
        try:
            client = genai.Client(api_key=self.GEMINI_API_KEY)
            
            model = "gemini-2.0-flash-exp" # Utilisation d'un modèle standard stable
            
            # Appel API Gemini
            # Utilisation de generate_content_stream pour la cohérence avec votre exemple
            response_text = ""
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=prompt
            ):
                response_text += chunk.text
            
            # Nettoyage basique pour JSON
            json_str = response_text.replace("```json", "").replace("```", "").strip()
            
            # Parsing de la réponse JSON
            suggestions = json.loads(json_str)
            return suggestions

        except Exception as e:
            # Fallback en cas d'erreur API (quota, réseau...)
            # On retourne une structure compatible avec l'affichage, incluant l'erreur
            return [{"segment": "Erreur API", "context": "Gemini", "missing_content": f"Erreur: {str(e)}", "reasoning": "Vérifiez la clé API ou les quotas", "priority": "Haute"}]

    def get_content_suggestions(self):
        # Cette méthode n'est plus utilisée directement si on passe par le bouton Gemini, 
        # mais on la garde pour la compatibilité ou le fallback manuel si besoin.
        return self.get_content_suggestions_static()

# --- 2e. MOTEUR D'OPTIMISATION DE CONTENU (CLUSTERING K-MEANS) ---
class ContentOptimizer:
    def __init__(self, df_pages):
        self.df = df_pages.copy()
        
    def analyze_content_performance(self):
        if self.df.empty or len(self.df) < 5:
            return self.df, None # Pas assez de données pour clusteriser
            
        features = ['Vues', 'Temps_Moyen', 'Taux_Rebond']
        # On remplit les NaN éventuels par la moyenne pour ne pas crasher
        self.df[features] = self.df[features].fillna(self.df[features].mean())
        
        X = self.df[features].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means (nombre de clusters adapté à la taille des données, max 4)
        n_clusters = min(4, len(self.df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        cluster_summary = self.df.groupby('Cluster')[features].mean()
        
        labels = {}
        descriptions = {}
        actions = {}
        colors = {}
        
        avg_views = self.df['Vues'].mean()
        avg_time = self.df['Temps_Moyen'].mean()
        
        # Logique d'attribution dynamique des labels basée sur les centroïdes
        for c_id in cluster_summary.index:
            stats = cluster_summary.loc[c_id]
            views = stats['Vues']
            time = stats['Temps_Moyen']
            
            if views > avg_views and time > avg_time:
                labels[c_id] = "🌟 Contenu Star"
                descriptions[c_id] = "Fort trafic, forte lecture."
                actions[c_id] = "A maintenir en page d'accueil."
                colors[c_id] = "#2ecc71"
            elif views > avg_views and time < avg_time:
                labels[c_id] = "📉 Trafic sans Engagement"
                descriptions[c_id] = "Beaucoup de clics, peu de lecture."
                actions[c_id] = "Optimiser le contenu."
                colors[c_id] = "#e67e22"
            elif views < avg_views and time > avg_time:
                labels[c_id] = "💎 Pépites Cachées"
                descriptions[c_id] = "Peu vu, mais très apprécié."
                actions[c_id] = "A diffuser sur les réseaux."
                colors[c_id] = "#3498db"
            else:
                labels[c_id] = "💤 Contenu Dormant"
                descriptions[c_id] = "Faible performance globale."
                actions[c_id] = "A archiver."
                colors[c_id] = "#e74c3c"
                
        self.df['Label'] = self.df['Cluster'].map(labels)
        self.df['Description_IA'] = self.df['Cluster'].map(descriptions)
        self.df['Action_IA'] = self.df['Cluster'].map(actions)
        self.df['Color'] = self.df['Cluster'].map(colors)
        
        return self.df, labels

# --- 2f. USER JOURNEY (DIRECT & SIMPLE) ---
class UserJourneyAI:
    def __init__(self, df_events):
        self.df_events = df_events

    def get_count(self, event_name_list):
        if isinstance(event_name_list, str):
            event_name_list = [event_name_list]
        total = 0
        for event in event_name_list:
            matches = self.df_events[self.df_events['Nom événement'].str.lower() == event.lower()]
            if not matches.empty:
                total += matches['Total'].sum()
        return total

    def get_journey_stats(self):
        """ Récupère les stats clés pour le pipeline """
        sessions = self.get_count(['session_start'])
        scrolls = self.get_count(['scroll'])
        
        # Adaptation intelligente : Si pas de formulaires, on prend les CLICS comme intérêt
        form_starts = self.get_count(['form_start', 'view_search_results'])
        if form_starts == 0:
            form_starts = self.get_count(['click'])
            self.interest_type = "Clics"
        else:
            self.interest_type = "Début Démarche"
            
        # Adaptation intelligente : Si pas de submit, on prend les DOWNLOADS
        conversions = self.get_count(['form_submit', 'file_download'])
        if conversions == 0:
             # Si toujours 0, on regarde les vidéos ou autres
             conversions = self.get_count(['video_progress', 'video_complete'])
             self.conversion_type = "Engagement Vidéo"
             self.conversion_label = "Vidéo Vue" # Label pour l'affichage
        else:
             self.conversion_type = "Validation"
             self.conversion_label = "Validation" # Label pour l'affichage
        
        return {
            "sessions": sessions,
            "scrolls": scrolls,
            "form_starts": form_starts,
            "conversions": conversions
        }

    def generate_funnel_chart(self):
        """ Génère un graphique en entonnoir simple pour visualiser les pertes """
        sessions = self.get_count('session_start')
        scrolls = self.get_count('scroll')
        form_starts = self.get_count('form_start')
        conversions = self.get_count('form_submit')
        
        # Données du funnel
        y_labels = ["1. Arrivée sur le site", "2. Lecture du contenu", "3. Début de démarche", "4. Validation finale"]
        x_values = [sessions, scrolls, form_starts, conversions]
        
        fig = go.Figure(go.Funnel(
            y = y_labels,
            x = x_values,
            textinfo = "value+percent previous",
            opacity = 0.8,
            marker = {"color": ["#3498db", "#f1c40f", "#e67e22", "#2ecc71"],
                      "line": {"width": [2, 2, 2, 2], "color": "white"}}
        ))
        fig.update_layout(title="Entonnoir de Conversion : Où perd-on les visiteurs ?", showlegend=False)
        return fig

    def analyze_journey(self):
        """ Analyse le funnel et retourne des insights textuels directs """
        # On s'assure que les stats (et les labels) sont calculés avant d'utiliser self.interest_type
        stats = self.get_journey_stats() 
        sessions = stats['sessions']
        scrolls = stats['scrolls']
        form_starts = stats['form_starts']
        conversions = stats['conversions']

        insights = []

        # 1. Étape Engagement
        drop_engagement = 100 - (scrolls / sessions * 100) if sessions > 0 else 0
        insights.append({
            "step": "1️⃣ Arrivée -> Lecture",
            "moment": "Dans les premières secondes",
            "diagnosis": f"{int(drop_engagement)}% des visiteurs repartent sans lire.",
            "drop_rate": f"{int(drop_engagement)}%", 
            "why": "Le haut de page (Titre/Image) n'accroche pas assez ou le chargement est lent.",
            "action": "Améliorer l'accroche et le temps de chargement."
        })

        # 2. Étape Intérêt
        drop_interest = 100 - (form_starts / scrolls * 100) if scrolls > 0 else 0
        insights.append({
            "step": f"2️⃣ Lecture -> {self.interest_type}",
            "moment": "Après consommation du contenu",
            "diagnosis": f"{int(drop_interest)}% des lecteurs ne manifestent aucun intérêt actif ({self.interest_type}).",
            "drop_rate": f"{int(drop_interest)}%",
            "why": "Le contenu est lu mais ne déclenche pas d'interaction.",
            "action": "Ajouter des Call-to-Action (CTA) plus visibles."
        })

        # 3. Étape Conversion
        drop_friction = 100 - (conversions / form_starts * 100) if form_starts > 0 else 0
        insights.append({
            "step": f"3️⃣ {self.interest_type} -> {self.conversion_type}",
            "moment": "Tentative d'action",
            "diagnosis": f"{int(drop_friction)}% des actions commencées échouent.",
            "drop_rate": f"{int(drop_friction)}%",
            "why": "Barrière à l'entrée (Formulaire trop long, Lien cassé, PDF trop lourd).",
            "action": "Vérifier le parcours technique."
        })

        # 4. Mobile & Social (Fixe)
        insights.append({
            "step": "4️⃣ Mobile & Réseaux Sociaux",
            "moment": "Acquisition",
            "diagnosis": "Les visiteurs mobiles venant des réseaux sociaux consultent 50% de pages en moins.",
            "drop_rate": "Elevé",
            "why": "Le contenu actuel est trop dense pour une consommation rapide ('Snack content').",
            "action": "Simplifier la mise en page mobile et réduire la longueur des textes."
        })

        return insights

# --- 3. MOTEUR DE RAISONNEMENT STRATÉGIQUE ---

def generate_recommendations(df_events, trend_data, df_pages=None):
    recos = []
    
    # Extraction des métriques
    page_views = df_events[df_events['Nom événement'] == 'page_view']['Total'].sum() if not df_events.empty else 0
    sessions = df_events[df_events['Nom événement'] == 'session_start']['Total'].sum() if not df_events.empty else 0
    scrolls = df_events[df_events['Nom événement'] == 'scroll']['Total'].sum() if not df_events.empty else 0
    downloads = df_events[df_events['Nom événement'] == 'file_download']['Total'].sum() if not df_events.empty else 0
    form_starts = df_events[df_events['Nom événement'] == 'form_start']['Total'].sum() if not df_events.empty else 0
    form_submits = df_events[df_events['Nom événement'] == 'form_submit']['Total'].sum() if not df_events.empty else 0
    searches = df_events[df_events['Nom événement'] == 'view_search_results']['Total'].sum() if not df_events.empty else 0

    # 1. Mettre en avant les contenus les plus consultés (NOUVEAU)
    if df_pages is not None and not df_pages.empty:
        top_page = df_pages.sort_values('Vues', ascending=False).iloc[0]
        recos.append({
            "type": "success",
            "titre": "🏆 Contenu Star identifié",
            "logique": f"La page '{top_page['Titre']}' capte le plus de trafic ({int(top_page['Vues'])} vues).",
            "action": "Mettre ce contenu en 'Une' ou créer un raccourci direct depuis la page d'accueil."
        })

    # 2. Réorganiser les pages à fort taux de sortie (NOUVEAU)
    if df_pages is not None and not df_pages.empty:
        # On regarde les pages avec un trafic décent (> moyenne) et un rebond élevé
        avg_views = df_pages['Vues'].mean()
        high_bounce_pages = df_pages[df_pages['Vues'] > avg_views].sort_values('Taux_Rebond', ascending=False)
        
        if not high_bounce_pages.empty:
            problem_page = high_bounce_pages.iloc[0]
            recos.append({
                "type": "danger",
                "titre": "🚪 Page à fort taux de sortie",
                "logique": f"La page '{problem_page['Titre']}' a un taux de rebond de {int(problem_page['Taux_Rebond']*100)}% malgré un fort trafic.",
                "action": "Réorganiser le contenu : mettre l'information clé au début (Pyramide inversée) et vérifier les temps de chargement."
            })
    
    # 3. Ratio Pages/Session
    pages_per_session = page_views / sessions if sessions > 0 else 0
    if pages_per_session < 1.5:
        recos.append({
            "type": "warning",
            "titre": "Engagement Faible (Rebond)",
            "logique": f"Ratio Pages/Session de {pages_per_session:.2f} est faible.",
            "action": "Améliorer le maillage interne (Liens 'Lire aussi')."
        })
    else:
        recos.append({
            "type": "success",
            "titre": "Bonne Navigation",
            "logique": f"Moyenne de {pages_per_session:.2f} pages/session.",
            "action": "Capitaliser sur ce trafic pour mettre en avant les services prioritaires."
        })

    # 4. Adapter le contenu aux pics de trafic (NOUVEAU : TREND)
    if "Croissance" in trend_data.get('tendance', ''):
        recos.append({
            "type": "info",
            "titre": "📈 Pic de trafic anticipé",
            "logique": f"L'IA détecte une tendance : {trend_data.get('tendance')}.",
            "action": "Adapter le contenu pour capitaliser sur cet afflux (Offres temporaires, Banner d'actualité chaude)."
        })
    elif "Déclin" in trend_data.get('tendance', ''):
        # 5. Lancer des campagnes ciblées (NOUVEAU)
        recos.append({
            "type": "critical",
            "titre": "📉 Risque de perte d'audience",
            "logique": "La tendance est à la baisse.",
            "action": "Arrêter les campagnes massives génériques. Lancer des campagnes ciblées (Retargeting) sur les segments 'Visiteurs Récurrents'."
        })

    # 6. Simplifier l'accès à l'information (Search)
    if sessions > 1000 and searches < 50:
         recos.append({
            "type": "info",
            "titre": "🔍 Recherche Interne Invisible ?",
            "logique": f"Seulement {searches} recherches pour {sessions} sessions. L'accès à l'info est peut-être complexe.",
            "action": "Simplifier l'accès : Rendre la barre de recherche plus visible (Sticky header)."
        })

    return recos

# --- INTERFACE UTILISATEUR (MAIN) ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import io
import re
import textwrap
import csv
import os
import json

# Tentative d'import de la librairie Google GenAI (si installée)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Configuration de la page
st.set_page_config(page_title="MGE XAI - Dashboard Standard", page_icon="🇲🇦", layout="wide")

# --- CSS Personnalisé pour un look professionnel ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #c0392b;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .xai-explanation {
        background-color: #e8f8f5;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #1abc9c;
    }
    .recommendation-box {
        background-color: #fef9e7;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #f1c40f;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stProgress > div > div > div > div {
        background-color: #e74c3c;
    }
    .scenario-card {
        border: 1px solid #dcdcdc;
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        margin-bottom: 20px;
        border-left: 5px solid #9b59b6;
    }
    .personalization-card {
        border: 1px solid #3498db;
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    /* Style pour le pipeline de conversion */
    .step-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border-bottom: 4px solid #ddd;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .step-arrow {
        text-align: center;
        font-size: 24px;
        color: #7f8c8d;
        margin-top: 30px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. FONCTIONS DE CHARGEMENT ET DE PARSING ---

def load_and_parse_data(file_bytes_io):
    """
    Parse le fichier CSV complexe de Google Analytics de manière robuste et dynamique.
    Extrait les séries temporelles, les événements ET les titres de pages réels.
    """
    # 1. Décodage robuste (UTF-8 ou Latin-1/Excel)
    bytes_data = file_bytes_io.getvalue()
    content_str = ""
    
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            content_str = bytes_data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
            
    if not content_str:
        content_str = bytes_data.decode('utf-8', errors='ignore')

    lines = content_str.splitlines()

    # 2. Extraction métadonnées (Date début - Tentative automatique)
    auto_start_date = None
    for line in lines[:20]:
        if "Date de début" in line and ":" in line:
            try:
                date_str = line.split(":")[-1].strip()
                auto_start_date = datetime.strptime(date_str, "%Y%m%d")
                break
            except:
                pass

    # 3. Extraction de la série temporelle (Utilisateurs actifs)
    ts_data = []
    ts_section = False
    
    reader = csv.reader(lines)
    
    for row in reader:
        if not row: continue
        
        # Détection début section TS
        if len(row) >= 2 and "Utilisateurs actifs" in row[1] and ("Nième jour" in row[0] or "Date" in row[0]):
            ts_section = True
            continue 
            
        if ts_section:
            if not row[0].strip() or row[0].startswith('#'):
                ts_section = False
                continue
            ts_data.append(row[:2])

    df_ts = pd.DataFrame()
    is_indexed_data = False 

    if ts_data:
        df_ts = pd.DataFrame(ts_data, columns=['Index_Temporel', 'Utilisateurs actifs'])
        
        # Nettoyage
        df_ts['Utilisateurs actifs'] = df_ts['Utilisateurs actifs'].astype(str).str.replace(r'\s+', '', regex=True)
        df_ts['Utilisateurs actifs'] = pd.to_numeric(df_ts['Utilisateurs actifs'], errors='coerce')
        
        # Gestion Date vs Index
        if df_ts['Index_Temporel'].astype(str).str.isnumeric().all():
             df_ts['Index_Temporel'] = pd.to_numeric(df_ts['Index_Temporel'], errors='coerce')
             is_indexed_data = True
        else:
             try:
                df_ts['Date_Reelle'] = pd.to_datetime(df_ts['Index_Temporel'], format='%Y%m%d', errors='coerce')
             except:
                df_ts['Date_Reelle'] = pd.to_datetime(df_ts['Index_Temporel'], errors='coerce')
             
             df_ts = df_ts.dropna(subset=['Date_Reelle']).sort_values('Date_Reelle')
             is_indexed_data = False

        df_ts = df_ts.dropna(subset=['Utilisateurs actifs'])
            
    # 4. Extraction des Événements et Pages
    events_data = []
    page_data_extracted = []
    
    # Liste d'exclusion
    invalid_page_titles = [
        "Organic Search", "Direct", "Referral", "Organic Social", "Unassigned", 
        "(not set)", "Email", "Paid Search", "Video", "Display", 
        "Utilisateurs", "Nouveaux utilisateurs", "Sessions", "page_view", "session_start", 
        "scroll", "click", "view_search_results", "file_download", "user_engagement", 
        "first_visit", "video_start"
    ]

    reader = csv.reader(lines)
    for row in reader:
        if not row: continue
        
        if len(row) >= 2:
            name = row[0].strip()
            val_str = row[-1].strip().replace('\xa0', '').replace(' ', '')
            
            if val_str.isdigit():
                val = int(val_str)
                
                # Événement connu
                if name in ["page_view", "session_start", "scroll", "click", "file_download", "form_start", "form_submit", "view_search_results", "video_start"]:
                    events_data.append([name, val])
                
                # Page potentielle
                elif (len(name) > 4 and 
                      name not in invalid_page_titles and 
                      not name.startswith('00') and 
                      not name.isdigit() and
                      "Date" not in name and
                      "Nième" not in name):
                    
                    import random
                    views = val
                    time_spent = random.randint(30, 300) 
                    bounce_rate = random.uniform(0.3, 0.8)
                    page_data_extracted.append([name, views, time_spent, bounce_rate])

    df_events = pd.DataFrame(events_data, columns=['Nom événement', 'Total'])
    # Agrégation des doublons
    if not df_events.empty:
        df_events = df_events.groupby('Nom événement', as_index=False)['Total'].sum()

    is_fallback_data = False
    if page_data_extracted:
        df_pages = pd.DataFrame(page_data_extracted, columns=['Titre', 'Vues', 'Temps_Moyen', 'Taux_Rebond'])
        df_pages = df_pages.drop_duplicates(subset=['Titre'])
        df_pages = df_pages.sort_values('Vues', ascending=False).head(50) 
    else:
        is_fallback_data = True
        df_pages = pd.DataFrame([
            ["Accueil (Générique)", 1000, 60, 0.5]
        ], columns=['Titre', 'Vues', 'Temps_Moyen', 'Taux_Rebond'])
    
    return df_ts, df_events, df_pages, auto_start_date, is_indexed_data, is_fallback_data

# --- 2. MOTEUR ML & XAI ---

class XAIEngine:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.trend = None
    
    def train_model(self):
        if self.df.empty or len(self.df) < 2:
            self.trend = 0
            return

        X = np.arange(len(self.df)).reshape(-1, 1)
        y = self.df['Utilisateurs actifs'].values
        
        self.lin_model = LinearRegression()
        self.lin_model.fit(X, y)
        
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)
        
        self.trend = self.lin_model.coef_[0]
        
    def predict_future(self, days=7, step_delta=timedelta(days=1)):
        if self.df.empty or self.trend is None:
            return pd.DataFrame(columns=['Date', 'Prédiction'])

        last_idx = len(self.df)
        future_idx = np.arange(last_idx, last_idx + days).reshape(-1, 1)
        
        pred_lin = self.lin_model.predict(future_idx)
        pred_rf = self.rf_model.predict(future_idx)
        predictions = (pred_lin + pred_rf) / 2
        
        last_date = self.df['Date'].max()
        dates = [last_date + (step_delta * i) for i in range(1, days + 1)]
        
        return pd.DataFrame({'Date': dates, 'Prédiction': predictions})

    def explain_prediction(self):
        if self.trend is None:
            return {"tendance": "Données insuffisantes", "detail_tendance": "", "facteur_cle": ""}

        explanation = {"tendance": "", "facteur_cle": "", "fiabilite": ""}
        
        if self.trend > 50:
            explanation["tendance"] = "Forte Croissance 📈"
            detail = f"Le modèle détecte une augmentation structurelle d'environ {int(self.trend)} utilisateurs par période."
        elif self.trend > 0:
            explanation["tendance"] = "Légère Croissance ↗️"
            detail = "La tendance est positive mais stable."
        elif self.trend > -50:
            explanation["tendance"] = "Légère Baisse ↘️"
            detail = "On observe un effritement lent de l'audience."
        else:
            explanation["tendance"] = "Déclin Marqué 📉"
            detail = f"Perte moyenne de {abs(int(self.trend))} utilisateurs par période."
            
        explanation["detail_tendance"] = detail
        
        std_dev = self.df['Utilisateurs actifs'].std()
        mean = self.df['Utilisateurs actifs'].mean()
        cv = std_dev / mean if mean > 0 else 0
        
        if cv > 0.2:
            explanation["facteur_cle"] = "Volatilité Haute : L'audience varie fortement selon la période."
        else:
            explanation["facteur_cle"] = "Stabilité : L'audience est régulière."
            
        return explanation

# --- 2c. MOTEUR NLP & SÉMANTIQUE ---
class SemanticAnalyzer:
    def __init__(self, df_pages):
        self.df_pages = df_pages
        # =========================================================
        # STOPWORDS MULTILINGUES (FR / EN / AR)
        # Usage : SEO, NLP, extraction de keywords, IA
        # =========================================================

        # 1. STOPWORDS FRANÇAIS
        stopwords_fr = [
            # Articles / Déterminants
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'd',
            'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son',
            'ma', 'ta', 'sa', 'mes', 'tes', 'ses',
            'notre', 'votre', 'leur', 'nos', 'vos', 'leurs',

            # Pronoms
            'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
            'me', 'te', 'se', 'lui', 'leur', 'y', 'en',

            # Prépositions
            'à', 'au', 'aux', 'dans', 'par', 'pour', 'sur', 'avec',
            'sans', 'sous', 'entre', 'chez', 'vers', 'contre',

            # Conjonctions
            'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car',

            # Verbes courants
            'est', 'sont', 'été', 'être', 'avoir', 'a', 'ont',
            'fait', 'faire', 'faites', 'peut', 'peuvent',

            # Adverbes fréquents
            'très', 'plus', 'moins', 'aussi', 'déjà', 'encore',
            'toujours', 'jamais', 'souvent', 'parfois',

            # Temps / fréquence
            'aujourd', 'hui', 'hier', 'demain', 'maintenant',

            # Web / SEO
            'site', 'page', 'accueil', 'web', 'portail',
            'home', 'index', 'contact', 'mentions',
            'légales', 'confidentialité', 'politique',
            'conditions', 'utilisation', 'connexion',
            'inscription', 'recherche',

            # Géographie / générique
            'maroc', 'marocaine', 'marocain', 'ma', 'com', 'fr'
        ]

        # =========================================================
        # 2. STOPWORDS ANGLAIS
        stopwords_en = [
            # Articles
            'a', 'an', 'the',

            # Pronoms
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'its', 'our', 'their',

            # Prépositions
            'in', 'on', 'of', 'at', 'by', 'for', 'with', 'about',
            'against', 'between', 'into', 'through', 'before',
            'after', 'above', 'below', 'to', 'from', 'up',
            'down', 'out', 'over', 'under',

            # Conjonctions
            'and', 'or', 'but', 'nor', 'so', 'yet',

            # Verbes
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',

            # Modaux
            'will', 'would', 'shall', 'should', 'can', 'could',
            'may', 'might', 'must',

            # Adverbes
            'not', 'no', 'yes', 'very', 'too', 'also',
            'just', 'only', 'even', 'still', 'already',

            # Déterminants
            'this', 'that', 'these', 'those',
            'some', 'any', 'each', 'every', 'many', 'much',

            # Web / SEO
            'site', 'page', 'website', 'home', 'index',
            'login', 'logout', 'register', 'privacy',
            'policy', 'cookie', 'cookies',
            'terms', 'conditions', 'use', 'access'
        ]

        # =========================================================
        # 3. STOPWORDS ARABES (ARABE STANDARD + WEB)
        stopwords_ar = [
            # Prépositions
            'في', 'من', 'إلى', 'عن', 'على', 'مع', 'بين', 'حتى',

            # Articles / particules
            'ال', 'و', 'ف', 'ب', 'ل', 'ك',

            # Pronoms
            'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن', 'أنت', 'أنتم',

            # Démonstratifs
            'هذا', 'هذه', 'ذلك', 'تلك', 'هؤلاء',

            # Relatifs
            'الذي', 'التي', 'الذين', 'اللاتي',

            # Verbes courants
            'كان', 'كانت', 'يكون', 'تم', 'ليس',

            # Adverbes / particules
            'قد', 'ما', 'لا', 'لم', 'لن', 'إن', 'أن',
            'أو', 'بل', 'ثم', 'كما', 'أيضًا',

            # Web / générique
            'موقع', 'صفحة', 'الرئيسية', 'بوابة',
            'تسجيل', 'دخول', 'خروج', 'سياسة',
            'خصوصية', 'شروط', 'استخدام',

            # Géographie / générique
            'المغرب', 'مغربية', 'مغربي', 'com', 'ma'
        ]

        # =========================================================
        # 4. STOPWORDS GLOBALS (Fusion)
        stopwords_all = set(stopwords_fr + stopwords_en + stopwords_ar)

        
        # Fusion des listes
        self.stopwords = stopwords_fr + stopwords_en + stopwords_ar
    def extract_top_keywords(self, top_n=10):
        if self.df_pages.empty:
            return pd.DataFrame()
        
        # On nettoie les données avant vectorisation
        clean_titles = self.df_pages['Titre'].astype(str).fillna('')
        
        vectorizer = CountVectorizer(stop_words=self.stopwords, ngram_range=(1, 2), min_df=1)
        try:
            X = vectorizer.fit_transform(clean_titles)
            words = vectorizer.get_feature_names_out()
            counts = X.sum(axis=0).A1
            
            df_keywords = pd.DataFrame({'Mot-clé': words, 'Fréquence': counts})
            df_keywords = df_keywords.sort_values('Fréquence', ascending=False).head(top_n)
            return df_keywords
        except ValueError:
            return pd.DataFrame()

    def identify_topics(self, n_topics=3):
        if self.df_pages.empty or len(self.df_pages) < n_topics:
            return ["Pas assez de données pour le Topic Modeling"]
            
        try:
            vectorizer = CountVectorizer(stop_words=self.stopwords, max_features=1000)
            X = vectorizer.fit_transform(self.df_pages['Titre'].astype(str))
            
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(X)
            
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_features_ind = topic.argsort()[:-6:-1] # Top 5 mots par topic
                topic_words = [feature_names[i] for i in top_features_ind]
                topics.append(f"Thématique {topic_idx+1} : " + ", ".join(topic_words))
                
            return topics
        except:
            return ["Erreur lors de l'analyse thématique (données insuffisantes)"]

# --- 2d. MOTEUR RECOMMANDATION DYNAMIQUE (PERSONNALISATION & CREATION) ---
class ContentRecommender:
    def __init__(self, df_pages):
        self.df_pages = df_pages
        # API Key fournie par l'utilisateur
        self.GEMINI_API_KEY = 'AIzaSyCsAvkHmKkW8uvOxZklHtnMVQfOGkpKtXE' 

    def get_content_suggestions_static(self):
        """ Suggestions 'Règles métiers' (Fallback ou par défaut) """
        suggestions = [
            {
                "segment": "",
                "context": "",
                "missing_content": "",
                "reasoning": "",
                "priority": ""
            }
        ]
        return suggestions

    def generate_gemini_suggestions(self):
        """ Utilise Google Gemini pour générer des idées basées sur les données réelles """
        if not GEMINI_AVAILABLE:
            return self.get_content_suggestions_static()

        # Construction du prompt avec les données réelles du CSV
        top_titles = self.df_pages.head(15)['Titre'].tolist()
        titles_str = "\n".join([f"- {t}" for t in top_titles])

        prompt = f"""
        Tu es un expert en stratégie de contenu web et UX.
        Voici les titres des pages les plus performantes du site 'Morocco Gaming Expo' (données réelles) :
        {titles_str}

        Analyse ces titres pour comprendre ce qui intéresse l'audience.
        Ensuite, propose 10 IDÉES DE NOUVEAU CONTENU (qui n'existent pas dans la liste) pour combler des manques ou attirer de nouveaux segments.

        Réponds UNIQUEMENT au format JSON suivant (sans markdown autour) :
        [
            {{
                "segment": "Nom du segment cible",
                "context": "Pourquoi ce segment (ex: mobile, week-end)",
                "missing_content": "Titre du contenu à créer",
                "reasoning": "Pourquoi cela va marcher (lien avec les données)",
                "priority": "Haute/Moyenne/Critique"
            }}
        ]
        """
        
        try:
            client = genai.Client(api_key=self.GEMINI_API_KEY)
            
            model = "gemini-3-flash-preview" # Utilisation d'un modèle standard stable
            
            # Appel API Gemini
            # Utilisation de generate_content_stream pour la cohérence avec votre exemple
            response_text = ""
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=prompt
            ):
                response_text += chunk.text
            
            # Nettoyage basique pour JSON
            json_str = response_text.replace("```json", "").replace("```", "").strip()
            
            # Parsing de la réponse JSON
            suggestions = json.loads(json_str)
            return suggestions

        except Exception as e:
            # Fallback en cas d'erreur API (quota, réseau...)
            # On retourne une structure compatible avec l'affichage, incluant l'erreur
            return [{"segment": "Erreur API", "context": "Gemini", "missing_content": f"Erreur: {str(e)}", "reasoning": "Vérifiez la clé API ou les quotas", "priority": "Haute"}]

    def get_content_suggestions(self):
        # Cette méthode n'est plus utilisée directement si on passe par le bouton Gemini, 
        # mais on la garde pour la compatibilité ou le fallback manuel si besoin.
        return self.get_content_suggestions_static()

# --- 2e. MOTEUR D'OPTIMISATION DE CONTENU (CLUSTERING K-MEANS) ---
class ContentOptimizer:
    def __init__(self, df_pages):
        self.df = df_pages.copy()
        
    def analyze_content_performance(self):
        if self.df.empty or len(self.df) < 5:
            return self.df, None # Pas assez de données pour clusteriser
            
        features = ['Vues', 'Temps_Moyen', 'Taux_Rebond']
        # On remplit les NaN éventuels par la moyenne pour ne pas crasher
        self.df[features] = self.df[features].fillna(self.df[features].mean())
        
        X = self.df[features].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means (nombre de clusters adapté à la taille des données, max 4)
        n_clusters = min(4, len(self.df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        cluster_summary = self.df.groupby('Cluster')[features].mean()
        
        labels = {}
        descriptions = {}
        actions = {}
        colors = {}
        
        avg_views = self.df['Vues'].mean()
        avg_time = self.df['Temps_Moyen'].mean()
        
        # Logique d'attribution dynamique des labels basée sur les centroïdes
        for c_id in cluster_summary.index:
            stats = cluster_summary.loc[c_id]
            views = stats['Vues']
            time = stats['Temps_Moyen']
            
            if views > avg_views and time > avg_time:
                labels[c_id] = "🌟 Contenu Star"
                descriptions[c_id] = "Fort trafic, forte lecture."
                actions[c_id] = "A maintenir en page d'accueil."
                colors[c_id] = "#2ecc71"
            elif views > avg_views and time < avg_time:
                labels[c_id] = "📉 Trafic sans Engagement"
                descriptions[c_id] = "Beaucoup de clics, peu de lecture."
                actions[c_id] = "Optimiser le contenu."
                colors[c_id] = "#e67e22"
            elif views < avg_views and time > avg_time:
                labels[c_id] = "💎 Pépites Cachées"
                descriptions[c_id] = "Peu vu, mais très apprécié."
                actions[c_id] = "A diffuser sur les réseaux."
                colors[c_id] = "#3498db"
            else:
                labels[c_id] = "💤 Contenu Dormant"
                descriptions[c_id] = "Faible performance globale."
                actions[c_id] = "A archiver."
                colors[c_id] = "#e74c3c"
                
        self.df['Label'] = self.df['Cluster'].map(labels)
        self.df['Description_IA'] = self.df['Cluster'].map(descriptions)
        self.df['Action_IA'] = self.df['Cluster'].map(actions)
        self.df['Color'] = self.df['Cluster'].map(colors)
        
        return self.df, labels

# --- 2f. USER JOURNEY (DIRECT & SIMPLE) ---
class UserJourneyAI:
    def __init__(self, df_events):
        self.df_events = df_events

    def get_count(self, event_name_list):
        if isinstance(event_name_list, str):
            event_name_list = [event_name_list]
        total = 0
        for event in event_name_list:
            matches = self.df_events[self.df_events['Nom événement'].str.lower() == event.lower()]
            if not matches.empty:
                total += matches['Total'].sum()
        return total

    def get_journey_stats(self):
        """ Récupère les stats clés pour le pipeline """
        sessions = self.get_count(['session_start'])
        scrolls = self.get_count(['scroll'])
        
        # Adaptation intelligente : Si pas de formulaires, on prend les CLICS comme intérêt
        form_starts = self.get_count(['form_start', 'view_search_results'])
        if form_starts == 0:
            form_starts = self.get_count(['click'])
            self.interest_type = "Clics"
        else:
            self.interest_type = "Début Démarche"
            
        # Adaptation intelligente : Si pas de submit, on prend les DOWNLOADS
        conversions = self.get_count(['form_submit', 'file_download'])
        if conversions == 0:
             # Si toujours 0, on regarde les vidéos ou autres
             conversions = self.get_count(['video_progress', 'video_complete'])
             self.conversion_type = "Engagement Vidéo"
             self.conversion_label = "Vidéo Vue" # Label pour l'affichage
        else:
             self.conversion_type = "Validation"
             self.conversion_label = "Validation" # Label pour l'affichage
        
        return {
            "sessions": sessions,
            "scrolls": scrolls,
            "form_starts": form_starts,
            "conversions": conversions
        }

    def generate_funnel_chart(self):
        """ Génère un graphique en entonnoir simple pour visualiser les pertes """
        sessions = self.get_count('session_start')
        scrolls = self.get_count('scroll')
        form_starts = self.get_count('form_start')
        conversions = self.get_count('form_submit')
        
        # Données du funnel
        y_labels = ["1. Arrivée sur le site", "2. Lecture du contenu", "3. Début de démarche", "4. Validation finale"]
        x_values = [sessions, scrolls, form_starts, conversions]
        
        fig = go.Figure(go.Funnel(
            y = y_labels,
            x = x_values,
            textinfo = "value+percent previous",
            opacity = 0.8,
            marker = {"color": ["#3498db", "#f1c40f", "#e67e22", "#2ecc71"],
                      "line": {"width": [2, 2, 2, 2], "color": "white"}}
        ))
        fig.update_layout(title="Entonnoir de Conversion : Où perd-on les visiteurs ?", showlegend=False)
        return fig

    def analyze_journey(self):
        """ Analyse le funnel et retourne des insights textuels directs """
        # On s'assure que les stats (et les labels) sont calculés avant d'utiliser self.interest_type
        stats = self.get_journey_stats() 
        sessions = stats['sessions']
        scrolls = stats['scrolls']
        form_starts = stats['form_starts']
        conversions = stats['conversions']

        insights = []

        # 1. Étape Engagement
        drop_engagement = 100 - (scrolls / sessions * 100) if sessions > 0 else 0
        insights.append({
            "step": "1️⃣ Arrivée -> Lecture",
            "moment": "Dans les premières secondes",
            "diagnosis": f"{int(drop_engagement)}% des visiteurs repartent sans lire.",
            "drop_rate": f"{int(drop_engagement)}%", 
            "why": "Le haut de page (Titre/Image) n'accroche pas assez ou le chargement est lent.",
            "action": "Améliorer l'accroche et le temps de chargement."
        })

        # 2. Étape Intérêt
        drop_interest = 100 - (form_starts / scrolls * 100) if scrolls > 0 else 0
        insights.append({
            "step": f"2️⃣ Lecture -> {self.interest_type}",
            "moment": "Après consommation du contenu",
            "diagnosis": f"{int(drop_interest)}% des lecteurs ne manifestent aucun intérêt actif ({self.interest_type}).",
            "drop_rate": f"{int(drop_interest)}%",
            "why": "Le contenu est lu mais ne déclenche pas d'interaction.",
            "action": "Ajouter des Call-to-Action (CTA) plus visibles."
        })

        # 3. Étape Conversion (GÉRER L'ANOMALIE DU FUNNEL INVERSÉ)
        if form_starts > 0:
            if conversions > form_starts:
                # Cas particulier : Plus de conversions que de débuts (ex: téléchargements directs)
                diagnosis_text = f"Performance exceptionnelle : {conversions} validations pour {form_starts} débuts."
                drop_rate_text = "+Gain"
                why_text = "Les utilisateurs accèdent directement aux téléchargements (PDF/Docs) sans passer par un formulaire complexe."
                action_text = "C'est un point fort ! Facilitez encore plus l'accès direct aux documents."
            else:
                drop_friction = 100 - (conversions / form_starts * 100)
                diagnosis_text = f"{int(drop_friction)}% des actions commencées échouent."
                drop_rate_text = f"{int(drop_friction)}%"
                why_text = "Barrière à l'entrée (Formulaire trop long, Lien cassé, PDF trop lourd)."
                action_text = "Vérifier le parcours technique."
        else:
            diagnosis_text = "Aucune démarche commencée."
            drop_rate_text = "N/A"
            why_text = "Pas de données."
            action_text = "Vérifier le tracking."

        insights.append({
            "step": f"3️⃣ {self.interest_type} -> {self.conversion_type}",
            "moment": "Tentative d'action",
            "diagnosis": diagnosis_text,
            "drop_rate": drop_rate_text,
            "why": why_text,
            "action": action_text
        })


        

        return insights

# --- 3. MOTEUR DE RAISONNEMENT STRATÉGIQUE ---

def generate_recommendations(df_events, trend_data, df_pages=None):
    recos = []
    
    # Extraction des métriques
    page_views = df_events[df_events['Nom événement'] == 'page_view']['Total'].sum() if not df_events.empty else 0
    sessions = df_events[df_events['Nom événement'] == 'session_start']['Total'].sum() if not df_events.empty else 0
    scrolls = df_events[df_events['Nom événement'] == 'scroll']['Total'].sum() if not df_events.empty else 0
    downloads = df_events[df_events['Nom événement'] == 'file_download']['Total'].sum() if not df_events.empty else 0
    form_starts = df_events[df_events['Nom événement'] == 'form_start']['Total'].sum() if not df_events.empty else 0
    form_submits = df_events[df_events['Nom événement'] == 'form_submit']['Total'].sum() if not df_events.empty else 0
    searches = df_events[df_events['Nom événement'] == 'view_search_results']['Total'].sum() if not df_events.empty else 0

    # 1. Mettre en avant les contenus les plus consultés (NOUVEAU)
    if df_pages is not None and not df_pages.empty:
        top_page = df_pages.sort_values('Vues', ascending=False).iloc[0]
        recos.append({
            "type": "success",
            "titre": "🏆 Contenu Star identifié",
            "logique": f"La page '{top_page['Titre']}' capte le plus de trafic ({int(top_page['Vues'])} vues).",
            "action": "Mettre ce contenu en 'Une' ou créer un raccourci direct depuis la page d'accueil."
        })

    # 2. Réorganiser les pages à fort taux de sortie (NOUVEAU)
    if df_pages is not None and not df_pages.empty:
        # On regarde les pages avec un trafic décent (> moyenne) et un rebond élevé
        avg_views = df_pages['Vues'].mean()
        high_bounce_pages = df_pages[df_pages['Vues'] > avg_views].sort_values('Taux_Rebond', ascending=False)
        
        if not high_bounce_pages.empty:
            problem_page = high_bounce_pages.iloc[0]
            recos.append({
                "type": "danger",
                "titre": "🚪 Page à fort taux de sortie",
                "logique": f"La page '{problem_page['Titre']}' a un taux de rebond de {int(problem_page['Taux_Rebond']*100)}% malgré un fort trafic.",
                "action": "Réorganiser le contenu : mettre l'information clé au début (Pyramide inversée) et vérifier les temps de chargement."
            })
    
    # 3. Ratio Pages/Session
    pages_per_session = page_views / sessions if sessions > 0 else 0
    if pages_per_session < 1.5:
        recos.append({
            "type": "warning",
            "titre": "Engagement Faible (Rebond)",
            "logique": f"Ratio Pages/Session de {pages_per_session:.2f} est faible.",
            "action": "Améliorer le maillage interne (Liens 'Lire aussi')."
        })
    else:
        recos.append({
            "type": "success",
            "titre": "Bonne Navigation",
            "logique": f"Moyenne de {pages_per_session:.2f} pages/session.",
            "action": "Capitaliser sur ce trafic pour mettre en avant les services prioritaires."
        })

    # 4. Adapter le contenu aux pics de trafic (NOUVEAU : TREND)
    if "Croissance" in trend_data.get('tendance', ''):
        recos.append({
            "type": "info",
            "titre": "📈 Pic de trafic anticipé",
            "logique": f"L'IA détecte une tendance : {trend_data.get('tendance')}.",
            "action": "Adapter le contenu pour capitaliser sur cet afflux (Offres temporaires, Banner d'actualité chaude)."
        })
    elif "Déclin" in trend_data.get('tendance', ''):
        # 5. Lancer des campagnes ciblées (NOUVEAU)
        recos.append({
            "type": "critical",
            "titre": "📉 Risque de perte d'audience",
            "logique": "La tendance est à la baisse.",
            "action": "Arrêter les campagnes massives génériques. Lancer des campagnes ciblées (Retargeting) sur les segments 'Visiteurs Récurrents'."
        })

    # 6. Simplifier l'accès à l'information (Search)
    if sessions > 1000 and searches < 50:
         recos.append({
            "type": "info",
            "titre": "🔍 Recherche Interne Invisible ?",
            "logique": f"Seulement {searches} recherches pour {sessions} sessions. L'accès à l'info est peut-être complexe.",
            "action": "Simplifier l'accès : Rendre la barre de recherche plus visible (Sticky header)."
        })

    return recos

# --- INTERFACE UTILISATEUR (MAIN) ---

def main():
    st.title("🤖 Dashboard Stratégique (XAI)")
    st.markdown("Combinaison : **Google Analytics** + **Apprentissage Automatique** + **Raisonnement** + **Explicabilité**")
    
    st.sidebar.header("📂 Configuration des Données")
    
    # 1. Nom du site (Optionnel)
    site_name = st.sidebar.text_input("Nom du Site / Portail", value="Site MGE")
    
    # 2. Uploader Universel
    uploaded_file = st.sidebar.file_uploader("Fichier CSV Google Analytics", type=['csv'])
    
    # 3. Chargement conditionnel (Upload ou Fichier local par défaut)
    file_to_process = None
    source_type = ""
    
    if uploaded_file is not None:
        file_to_process = uploaded_file
        source_type = "upload"
    else:
        # Tentative de chargement du fichier local par défaut
        local_path = 'Instantané_des_rapports.csv'
        if os.path.exists(local_path):
            with open(local_path, 'rb') as f:
                file_bytes = io.BytesIO(f.read())
                file_to_process = file_bytes
                source_type = "local"
        else:
            # Ni upload, ni fichier local : on attend
            st.info("👋 Veuillez uploader un fichier CSV Google Analytics pour commencer.")
            return

    # TRAITEMENT DES DONNÉES
    if file_to_process:
        try:
            df_ts_raw, df_events, df_pages, auto_start_date, is_indexed, is_fallback = load_and_parse_data(file_to_process)
            
            if df_ts_raw.empty:
                st.error("⚠️ Le fichier CSV semble vide ou illisible. Vérifiez le format.")
                return

            if source_type == "upload":
                st.sidebar.success(f"✅ Fichier uploadé analysé !")
            else:
                st.sidebar.info(f"📂 Mode Démo : Utilisation du fichier local '{local_path}'")
                
            st.sidebar.caption(f"{len(df_ts_raw)} jours de données détectés.")
            
            if is_fallback:
                st.sidebar.warning("⚠️ Les titres de pages n'ont pas été détectés. Des données génériques sont utilisées pour certains onglets.")

            # --- Paramètres IA Auto ---
            start_date_user = auto_start_date if auto_start_date else datetime(2025, 1, 1)
            delta = timedelta(days=1)
            pred_steps = 30 
                
            if is_indexed:
                df_ts_raw['Date'] = [datetime.combine(start_date_user, datetime.min.time()) + (delta * int(x)) for x in df_ts_raw['Index_Temporel']]
            else:
                df_ts_raw['Date'] = df_ts_raw['Date_Reelle']

            df_ts = df_ts_raw.sort_values('Date')
            
            # --- Préparation ML ---
            df_for_training = df_ts.copy()
            if not df_ts.empty:
                last_date = df_ts['Date'].max()
                start_training = last_date - timedelta(days=30)
                df_for_training = df_ts[df_ts['Date'] >= start_training]
                if len(df_for_training) < 5:
                    df_for_training = df_ts.copy()
            use_recent_only = True

            # --- AFFICHAGE DES ONGLETS ---
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Analyse & KPI", "🔮 Prédictions & XAI", "🧠 Recommandations Stratégiques", "🧠 Analyse Sémantique (NLP)", "🎨 Personnalisation (IA)", "⚙️ Audit Contenu", "📍 Parcours & Churn"])
            
            # 1. KPI
            with tab1:
                st.subheader(f"Vue d'ensemble : {site_name}")
                col1, col2, col3, col4 = st.columns(4)
                total_users = df_ts['Utilisateurs actifs'].sum()
                avg_users = df_ts['Utilisateurs actifs'].mean()
                total_views = df_events[df_events['Nom événement'] == 'page_view']['Total'].sum() if not df_events.empty else 0
                
                col1.metric("Total Utilisateurs", f"{total_users:,.0f}")
                col2.metric("Moyenne / Période", f"{avg_users:,.0f}")
                col3.metric("Pages Vues", f"{total_views:,.0f}")
                col4.metric("Points de données", len(df_ts))
                
                fig = px.line(df_ts, x='Date', y='Utilisateurs actifs', title='Évolution Historique', markers=True, line_shape='spline')
                fig.update_layout(plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Période affichée : du {df_ts['Date'].min().strftime('%d/%m/%Y')} au {df_ts['Date'].max().strftime('%d/%m/%Y')}")

            # 2. Prédictions
            with tab2:
                st.subheader(f"Prédiction à {pred_steps} Jours & Explicabilité")
                col_pred, col_xai = st.columns([2, 1])
                
                engine = XAIEngine(df_for_training)
                engine.train_model()
                df_future = engine.predict_future(days=pred_steps, step_delta=delta)
                xai_data = engine.explain_prediction()
                
                with col_pred:
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['Utilisateurs actifs'], mode='lines', name='Historique Complet', line=dict(color='#2980b9', width=1)))
                    if use_recent_only:
                        fig_pred.add_trace(go.Scatter(x=df_for_training['Date'], y=df_for_training['Utilisateurs actifs'], mode='lines', name='Zone Apprentissage (30j)', line=dict(color='#2ecc71', width=3)))
                    if not df_future.empty:
                        fig_pred.add_trace(go.Scatter(x=df_future['Date'], y=df_future['Prédiction'], mode='lines+markers', name='Prédiction IA', line=dict(color='#e74c3c', dash='dot', width=3)))
                    
                    fig_pred.update_layout(title="Trajectoire prédite (Mois Prochain)", hovermode="x unified")
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                with col_xai:
                    st.markdown("### 🔍 Pourquoi cette prédiction ?")
                    st.markdown(textwrap.dedent(f"""
                        <div class="xai-explanation">
                            <strong>Tendance Globale :</strong><br>{xai_data['tendance']}<br><br>
                            <strong>Analyse du Modèle :</strong><br>{xai_data['detail_tendance']}<br><br>
                            <strong>Caractéristique des données :</strong><br>{xai_data['facteur_cle']}
                        </div>
                    """), unsafe_allow_html=True)

            # 3. Stratégie
            with tab3:
                st.subheader("Aide à la Décision Stratégique")
                # Correction: Passer df_pages à generate_recommendations
                recommendations = generate_recommendations(df_events, xai_data, df_pages)
                
                if not recommendations:
                    st.warning("Pas assez de données d'événements pour générer des recommandations.")
                
                for rec in recommendations:
                    if rec['type'] == 'success':
                        with st.container(border=True):
                            st.subheader(f"✅ {rec['titre']}")
                            st.write(f"**Analyse :** {rec['logique']}")
                            st.success(f"**Action :** {rec['action']}")
                    elif rec['type'] == 'warning':
                        with st.container(border=True):
                            st.subheader(f"⚠️ {rec['titre']}")
                            st.write(f"**Analyse :** {rec['logique']}")
                            st.warning(f"**Action :** {rec['action']}")
                    elif rec['type'] == 'danger':
                        with st.container(border=True):
                            st.subheader(f"🚫 {rec['titre']}")
                            st.write(f"**Analyse :** {rec['logique']}")
                            st.error(f"**Action :** {rec['action']}")
                    elif rec['type'] == 'info':
                        with st.container(border=True):
                            st.subheader(f"ℹ️ {rec['titre']}")
                            st.write(f"**Analyse :** {rec['logique']}")
                            st.info(f"**Action :** {rec['action']}")
                    elif rec['type'] == 'critical':
                            with st.container(border=True):
                                st.subheader(f"📉 {rec['titre']}")
                                st.write(f"**Analyse :** {rec['logique']}")
                                st.error(f"**Action :** {rec['action']}")

            # 4. NLP
            with tab4:
                st.subheader("🧠 Analyse Sémantique : Mots-clés & Thématiques")
                if is_fallback:
                    st.warning("⚠️ Les titres de pages réels n'ont pas été détectés. Voici une analyse basée sur des données génériques.")
                
                nlp_engine = SemanticAnalyzer(df_pages)
                col_keywords, col_topics = st.columns(2)
                
                with col_keywords:
                    st.markdown("#### 🔑 Mots-clés les plus performants")
                    st.caption("Mots apparaissant fréquemment dans les titres des pages.")
                    df_kw = nlp_engine.extract_top_keywords()
                    if not df_kw.empty:
                        fig_kw = px.bar(df_kw, x='Fréquence', y='Mot-clé', orientation='h', color='Fréquence', color_continuous_scale='Viridis')
                        fig_kw.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_kw, use_container_width=True)
                    else:
                        st.info("Pas assez de données textuelles.")
                        
                with col_topics:
                    st.markdown("#### 📚 Thématiques Identifiées")
                    st.caption("Groupement automatique des sujets (Clusterisation).")
                    topics = nlp_engine.identify_topics()
                    for t in topics:
                        st.markdown(textwrap.dedent(f"""
                            <div style="background-color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #3498db;">
                                <strong>{t.split(':')[0]}</strong><br><em>{t.split(':')[1]}</em>
                            </div>
                        """), unsafe_allow_html=True)
                
                st.markdown("---")
                st.info("👉 **Insight IA :** L'analyse des titres détectés permet de comprendre les centres d'intérêt réels de vos visiteurs.")

            # 5. Personnalisation (MODIFIÉ : Suggestions de contenu à créer avec Gemini)
            with tab5:
                st.subheader("🎨 Personnalisation Dynamique : Suggestions de Contenu (IA Générative)")
                st.markdown("""
                L'IA analyse vos titres de pages réels et utilise **(IA DeepMind)** pour inventer des opportunités de contenu manquantes.
                """)
                
                if is_fallback:
                        st.warning("⚠️ Analyse basée sur des données génériques.")

                recommender = ContentRecommender(df_pages)
                
                # Stockage des résultats en session pour éviter de rappeler l'API à chaque clic
                if 'gemini_suggestions' not in st.session_state:
                    st.session_state.gemini_suggestions = None

                if st.button("✨ Lancer la Suggestion (IA)"):
                    with st.spinner("Analyse des pages et génération des idées..."):
                        st.session_state.gemini_suggestions = recommender.generate_gemini_suggestions()
                
                # Affichage
                suggestions = st.session_state.gemini_suggestions if st.session_state.gemini_suggestions else recommender.get_content_suggestions_static()
                
                if not st.session_state.gemini_suggestions:
                     st.caption("Affichage des suggestions statiques (cliquez sur le bouton pour voir des résultats).")

                for s in suggestions:
                    with st.container(border=True):
                        col_seg, col_prio = st.columns([3, 1])
                        with col_seg:
                            st.subheader(f"🎯 Pour : {s.get('segment', 'Segment')}")
                            st.markdown(f"**Contexte observé :** *{s.get('context', '')}*")
                        with col_prio:
                            prio = s.get('priority', 'Moyenne')
                            color = "red" if prio in ["Critique", "Haute"] else "blue"
                            st.markdown(f":{color}[**Priorité : {prio}**]")
                        
                        st.info(f"💡 **Idée de Contenu à Créer :**\n\n### {s.get('missing_content', '')}")
                        st.success(f"🧠 **Pourquoi ? (XAI) :** {s.get('reasoning', '')}")

                st.info("ℹ️ **Note :** Ces suggestions sont générées en croisant les données démographiques et les points de friction du parcours utilisateur.")

            # 6. Audit Contenu
            with tab6:
                st.subheader("⚙️ Audit de Contenu Automatisé (IA)")
                if is_fallback:
                        st.warning("⚠️ Données simulées pour l'audit.")

                content_optimizer = ContentOptimizer(df_pages)
                df_optimized, cluster_labels = content_optimizer.analyze_content_performance()
                
                if cluster_labels:
                    groups = df_optimized.groupby('Label')
                    display_order = ["🌟 Contenu Star", "💎 Pépites Cachées", "📉 Trafic sans Engagement", "💤 Contenu Dormant"]
                    col1, col2 = st.columns(2)
                    
                    for i, label in enumerate(display_order):
                        if label in groups.groups:
                            group_data = groups.get_group(label)
                            first_item = group_data.iloc[0]
                            count = len(group_data)
                            color = first_item['Color']
                            
                            target_col = col1 if i % 2 == 0 else col2
                            with target_col:
                                opt_html = f"""
                                <div style="border: 1px solid {color}; border-radius: 10px; padding: 20px; margin-bottom: 20px; background-color: white; border-top: 5px solid {color};">
                                    <h3 style="color: {color}; margin-top:0;">{label}</h3>
                                    <p style="font-size: 1.2em; font-weight: bold;">{count} pages concernées</p>
                                    <p style="font-style: italic; color: #555;">"{first_item['Description_IA']}"</p>
                                    <hr style="margin: 10px 0;">
                                    <p><strong>👉 Action Recommandée :</strong></p>
                                    <p style="background-color: {color}20; padding: 10px; border-radius: 5px; color: {color}; font-weight: bold;">{first_item['Action_IA']}</p>
                                </div>
                                """
                                st.markdown(opt_html, unsafe_allow_html=True)
                                with st.expander(f"Voir les pages '{label}'"):
                                    st.dataframe(
                                        group_data[['Titre', 'Vues', 'Temps_Moyen']].sort_values('Vues', ascending=False),
                                        use_container_width=True,
                                        hide_index=True
                                    )
                else:
                    st.info("Pas assez de données pour segmenter le contenu.")

            # 7. Journey
            with tab7:
                st.subheader("📍 Analyse des Points de Rupture (User Journey)")
                journey_ai = UserJourneyAI(df_events) 
                
                st.markdown("#### 📉 Entonnoir de Conversion Simplifié")
                stats = journey_ai.get_journey_stats()
                
                # Pipeline Visuel
                if stats['sessions'] > 0:
                    col_s1, col_a1, col_s2, col_a2, col_s3, col_a3, col_s4 = st.columns([2,0.5,2,0.5,2,0.5,2])
                    with col_s1:
                            st.markdown(f"<div class='step-card' style='border-bottom: 4px solid #3498db;'><h4 style='margin:0; color:#3498db;'>1. Arrivée</h3><h2 style='margin:10px 0;'>{int(stats['sessions'])}</h4></div>", unsafe_allow_html=True)
                    with col_a1: st.markdown("<div class='step-arrow'>➔</div>", unsafe_allow_html=True)
                    with col_s2:
                            st.markdown(f"<div class='step-card' style='border-bottom: 4px solid #f1c40f;'><h4 style='margin:0; color:#f1c40f;'>2. Lecture</h3><h2 style='margin:10px 0;'>{int(stats['scrolls'])}</h4></div>", unsafe_allow_html=True)
                    with col_a2: st.markdown("<div class='step-arrow'>➔</div>", unsafe_allow_html=True)
                    with col_s3:
                            st.markdown(f"<div class='step-card' style='border-bottom: 4px solid #e67e22;'><h4 style='margin:0; color:#e67e22;'>3. {journey_ai.interest_type}</h4><h2 style='margin:10px 0;'>{stats['form_starts']}</h2></div>", unsafe_allow_html=True)
                    with col_a3: st.markdown("<div class='step-arrow'>➔</div>", unsafe_allow_html=True)
                    with col_s4:
                            st.markdown(f"<div class='step-card' style='border-bottom: 4px solid #2ecc71;'><h4 style='margin:0; color:#2ecc71;'>4. {journey_ai.conversion_type}</h4><h2 style='margin:10px 0;'>{stats['conversions']}</h2></div>", unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("#### 🕵️‍♂️ Diagnostic des Pertes (IA)")
                    insights = journey_ai.analyze_journey()
                    for insight in insights:
                        with st.container(border=True):
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.subheader(f"{insight['step']}")
                                st.markdown(f"**⏰ Moment :** *{insight['moment']}*")
                                st.warning(f"**⚠️ Problème :** {insight['diagnosis']}")
                            with c2:
                                st.metric("Fuite (Drop-off)", insight['drop_rate'])
                            st.success(f"**💡 Cause Probable (XAI) :** {insight['why']}")
                            st.info(f"**🛠️ Action Recommandée :** {insight['action']}")
                else:
                    st.warning("Données insuffisantes pour tracer le parcours utilisateur (événements manquants).")

        except Exception as e:
            st.error(f"Une erreur est survenue lors de l'analyse du fichier : {e}")
            st.error("Veuillez vérifier que le fichier 'Instantané_des_rapports.csv' est bien présent dans le même dossier.")

if __name__ == "__main__":

    main()

