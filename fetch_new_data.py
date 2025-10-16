import pandas as pd
import requests
import os
import time 
from config import NEWS_API_KEY 
from datetime import datetime, timedelta

# --- API Configuration ---
NEWSDATA_API_URL = "https://newsdata.io/api/1/news" 

# Define the timeframe: 5 months ago to now
five_months_ago = (datetime.now() - timedelta(days=5*30)).strftime('%Y-%m-%d')
today = datetime.now().strftime('%Y-%m-%d')

# List of reputable sources/domains (still useful for manual checks later)
REAL_SOURCES = [
    'bbc.co.uk', 'reuters.com', 'nytimes.com', 'cnn.com', 
    'washingtonpost.com', 'theguardian.com', 'apnews.com', 
    'timesofindia.indiatimes.com', 'deccanherald.com'
]

def fetch_recent_real_news(total_articles=100): 
    """Fetches real news articles using NewsData.io API."""
    
    if len(NEWS_API_KEY) < 20: 
        print("ERROR: API Key appears invalid or missing. Please verify config.py.")
        return pd.DataFrame()

    articles_list = []
    next_page_token = None
    
    print(f"Fetching up to {total_articles} articles using NewsData.io...")

    while len(articles_list) < total_articles:
        params = {
            'apikey': NEWS_API_KEY, 
            'q': 'technology OR finance OR politics OR health', 
            'language': 'en',
            'country': 'us,gb,in', 
            'size': 10,
        }
        
        if next_page_token:
            params['page'] = next_page_token
            
        try:
            response = requests.get(NEWSDATA_API_URL, params=params, timeout=15)
            response.raise_for_status() 
            data = response.json()
            
            if data.get('status') == 'error':
                 print(f"NewsData.io Error: {data.get('message')}")
                 break
            
            new_articles_count = 0
            for article in data.get('results', []):
                # We only keep articles with both a title and content
                if article['title'] and article['content']:
                    
                    # --- FIX APPLIED: SOURCE FILTERING REMOVED TO COLLECT ALL DATA ---
                    
                    full_text = f"{article['title']}. {article['content']}"
                    articles_list.append({
                        'text': full_text,
                        'label': 1  
                    })
                    new_articles_count += 1

            print(f"Page fetched. Collected {new_articles_count} new articles. Total: {len(articles_list)}")
            
            next_page_token = data.get('nextPage')
            if not next_page_token or len(data.get('results', [])) == 0:
                print("End of available pages reached.")
                break
                
            time.sleep(1) 

        except requests.exceptions.RequestException as e:
            print(f"API Request failed: {e}")
            break
            
    return pd.DataFrame(articles_list)

if __name__ == '__main__':
    new_df = fetch_recent_real_news(total_articles=100) 
    
    if not new_df.empty:
        new_df.to_csv('new_real_news_data.csv', index=False)
        print(f"\nSuccessfully collected and saved {len(new_df)} new real news articles to 'new_real_news_data.csv'.")
        print("NEXT: Run 'python train_and_save_model.py'")
    else:
        print("\nFailed to collect any new data. Check your NewsData.io account/plan limits.")