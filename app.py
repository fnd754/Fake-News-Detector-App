import joblib
import requests
import pandas as pd
import re
import os
from flask import Flask, render_template, request, redirect, url_for
from newspaper import Article
from forms import NewsForm
from config import NEWS_API_KEY, NEWS_API_URL

# --- Application Setup ---
app = Flask(__name__)
# IMPORTANT: Provide a strong secret key for production security
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secure_key_for_dev') 

# Load the trained model and vectorizer
try:
    model = joblib.load('model.pkl')
    tfidfvect = joblib.load('tfidfvect.pkl')
except FileNotFoundError:
    print("Error: Model or Vectorizer files not found. Run 'train_and_save_model.py' first.")
    model = None
    tfidfvect = None

# --- NEW: Text Cleaning Utility (Crucial for Prediction Consistency) ---
def clean_input_text(text):
    """Normalizes text for consistent training/prediction."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    text = ' '.join(text.split())
    return text

# --- Helper Functions (Unified for NewsData.io) ---

def get_article_content(url):
    """Fetches the URL and extracts clean article text and title."""
    try:
        article = Article(url, fetch_images=False, memoize_articles=False)
        # Add User-Agent header for fetching stability
        article.download(headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0)'})
        article.parse()
        
        article_text = f"{article.title}. {article.text}"
        
        if len(article_text) < 50 or not article.title:
             return None, None 

        return article_text, article.title
        
    except Exception as e:
        print(f"Error extracting content from URL {url}: {e}")
        return None, None

def check_external_sources(title):
    """Searches external news sources using the NewsData.io API for corroboration."""
    
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

# FIX: Modified to use the new cleaning function
def predict_news(news_text):
    """Vectorizes the text and predicts using the loaded ML model."""
    if model is None or tfidfvect is None:
        return "Model Error: ML components not loaded correctly."
        
    # --- CRITICAL FIX: Clean the input text before transformation ---
    cleaned_text = clean_input_text(news_text)
    
    # Vectorize the input text
    news_vect = tfidfvect.transform([cleaned_text])
    prediction = model.predict(news_vect)[0]
    
    return "REAL" if prediction == 1 else "FAKE"

def fetch_top_headlines():
    """Fetches a list of recent headlines using the NewsData.io API."""
    
    if len(NEWS_API_KEY) < 20: 
        return [{"title": "API Key Missing/Invalid", "url": "#", "description": "Please verify NEWS_API_KEY in config.py."}]

    params = {
        'apikey': NEWS_API_KEY, 
        'q': 'technology OR finance OR politics', # Broad query for variety
        'language': 'en',
        'country': 'us,gb,in',  
        'size': 10, 
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # NewsData.io returns articles under the 'results' key
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
    
    # Handle GET request from Live News Feed selection
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

    # Handle POST request from URL/Text submission form
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
    
    # Handle POST request from article selection form (redirects to index)
    if request.method == 'POST':
        selected_url = request.form.get('selected_url')
        if selected_url:
            # Redirect to index with the URL as a query parameter
            return redirect(url_for('index', url=selected_url))
            
    # Handle GET request (displays the feed)
    articles = fetch_top_headlines()
    
    return render_template('live_news.html', articles=articles)

if __name__ == '__main__':
    # Use environment variable for port in production
    port = int(os.environ.get('PORT', 5000)) 
    app.run(debug=True, host='0.0.0.0', port=port)