import joblib
import requests
import pandas as pd
import re
import os
from flask import Flask, render_template, request, redirect, url_for
from forms import NewsForm
from config import NEWS_API_KEY, NEWS_API_URL
# FIX: Switched from newspaper to goose3 for robust scraping
from goose3 import Goose 

# --- Application Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secure_key_for_dev') 

# Load the trained model and vectorizer
try:
    model = joblib.load('model.pkl')
    tfidfvect = joblib.load('tfidfvect.pkl')
except FileNotFoundError:
    print("Error: Model or Vectorizer files not found. Run 'train_and_save_model.py' first.")
    model = None
    tfidfvect = None

# --- Text Cleaning Utility (Crucial for Prediction Consistency) ---
def clean_input_text(text):
    """Normalizes input text for consistent training/prediction."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    text = ' '.join(text.split())
    return text

# --- Helper Functions ---

# FIX: Rewritten to use Goose3
def get_article_content(url):
    """Fetches the URL and extracts clean article text and title using Goose3."""
    try:
        # Initialize Goose3 with a browser-like User-Agent for better fetching success
        g = Goose({'browser_user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        
        # Extract the article
        article = g.extract(url=url)
        
        # Use Goose's cleaned text property
        article_text = f"{article.title}. {article.cleaned_text}"
        
        if len(article_text) < 50 or not article.title:
             return None, None 

        return article_text, article.title
        
    except Exception as e:
        print(f"Error extracting content from URL {url}: {e}")
        return None, None

def check_external_sources(title):
    # ... (Keep this function the same) ...
    if len(NEWS_API_KEY) < 20: 
        return "API Check Skipped (API Key invalid or missing)."

    params = {
        'apikey': NEWS_API_KEY,
        'q': title, 
        'language': 'en',
        'country': 'us,gb,in',
        'size': 5,
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=8)
        response.raise_for_status() 
        data = response.json()
        
        total_results = data.get('totalResults', 0)
        
        if total_results > 5:
            return f"SUCCESS: {total_results} similar articles found on external APIs. (High Corroboration)"
        elif total_results > 0:
            return f"NOTICE: {total_results} similar articles found. (Low Corroboration)"
        else:
            return "WARNING: Zero similar articles found on external APIs. (Lacks Corroboration)"
            
    except requests.exceptions.RequestException as e:
        return f"API Check Failed: Network or API error. ({e})"

def predict_news(news_text):
    """Vectorizes the text and predicts using the loaded ML model."""
    if model is None or tfidfvect is None:
        return "Model Error: ML components not loaded correctly."
        
    cleaned_text = clean_input_text(news_text)
    
    news_vect = tfidfvect.transform([cleaned_text])
    prediction = model.predict(news_vect)[0]
    
    return "REAL" if prediction == 1 else "FAKE"

def fetch_top_headlines():
    # ... (Keep this function the same, it uses the NewsData.io structure) ...
    if len(NEWS_API_KEY) < 20: 
        return [{"title": "API Key Missing/Invalid", "url": "#", "description": "Please verify NEWS_API_KEY in config.py."}]

    params = {
        'apikey': NEWS_API_KEY, 
        'q': 'technology OR finance OR politics', 
        'language': 'en',
        'country': 'us,gb,in',  
        'size': 10, 
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return data.get('results', [])
        
    except requests.exceptions.RequestException as e:
        return [{"title": "API Fetch Error", "url": "#", "description": f"Failed to fetch headlines: {e}"}]

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    form = NewsForm()
    prediction = None
    article_text = None
    user_url = None
    api_check_result = None
    source_type = None
    
    # ... (All routing logic remains the same, as it calls the fixed helper functions) ...
    if request.method == 'GET' and 'url' in request.args:
        user_url = request.args.get('url')
        form.url.data = user_url
        
        article_text, article_title = get_article_content(user_url)
        source_type = "URL"
        
        if article_text:
            prediction = predict_news(article_text)
            api_check_result = check_external_sources(article_title)
        else:
            prediction = "⚠️ Error: Could not process content from URL selected from feed."

    elif form.validate_on_submit():
        
        if form.url.data:
            user_url = form.url.data
            article_text, article_title = get_article_content(user_url)
            source_type = "URL"
            
        elif form.text.data:
            user_text = form.text.data
            article_text = user_text
            article_title = user_text[:50] 
            user_url = None
            source_type = "TEXT"
            
        else:
            prediction = "Error: Please enter a URL or news text."
            source_type = None

        if article_text and source_type:
            prediction = predict_news(article_text) 
            
            if source_type == "URL":
                 api_check_result = check_external_sources(article_title)
            elif source_type == "TEXT":
                 api_check_result = "API Check Skipped (Direct text input used)."
            
        elif source_type:
            prediction = f"⚠️ Error: Could not process content from {source_type}."
            
    return render_template('index.html', 
                           form=form, 
                           prediction=prediction, 
                           article_text=article_text, 
                           user_url=user_url,
                           api_check_result=api_check_result)

@app.route('/live_news_feed', methods=['GET', 'POST'])
def live_news_feed():
    
    if request.method == 'POST':
        selected_url = request.form.get('selected_url')
        if selected_url:
            return redirect(url_for('index', url=selected_url))
            
    articles = fetch_top_headlines()
    
    return render_template('live_news.html', articles=articles)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000)) 
    app.run(debug=True, host='0.0.0.0', port=port)