import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import os
import re

# FIX: Replicate the cleaning function used in app.py for consistency
def clean_input_text(text):
    """Normalizes text for consistent training/prediction."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    text = ' '.join(text.split())
    return text


print("--- Starting Model Retraining with ALL Datasets ---")

# --- 1. Define Data Sources ---
DATA_FILES = [
    {'file': 'random_dataset.csv', 'label_override': None},
    {'file': 'manual_fake_data.csv', 'label_override': 0}, 
    {'file': 'manual_real_data.csv', 'label_override': 1}, 
    {'file': 'new_real_news_data.csv', 'label_override': 1}, 
]

all_data = []

# --- 2. Load and Combine All Data ---
for source in DATA_FILES:
    file_path = source['file']
    label_override = source['label_override']
    
    if os.path.exists(file_path):
        try:
            # Use on_bad_lines='skip' to ignore malformed rows in manual files
            df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8') 
            
            if 'text' not in df.columns:
                print(f"Skipping {file_path}: Missing 'text' column.")
                continue
            
            if label_override is not None:
                 df['label'] = label_override
            
            if 'label' not in df.columns:
                 print(f"Skipping {file_path}: No 'label' column and no override provided.")
                 continue

            all_data.append(df)
            print(f"Loaded {len(df)} rows from {file_path}.")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    elif source['file'] == 'random_dataset.csv':
        print("CRITICAL ERROR: 'random_dataset.csv' not found. Cannot train model.")
        exit()
    else:
        print(f"Skipping {file_path}: File not found.")

if not all_data:
    print("FATAL ERROR: No training data could be loaded.")
    exit()
    
df_combined = pd.concat(all_data, ignore_index=True)
print(f"\nCombined dataset size BEFORE cleanup: {len(df_combined)} rows.")


# --- 3. CRITICAL FIX: Data Cleaning and Type Conversion ---
rows_before_drop = len(df_combined)
df_combined.dropna(subset=['text', 'label'], inplace=True)
rows_after_drop = len(df_combined)

if rows_before_drop > rows_after_drop:
    print(f"CLEANUP: Dropped {rows_before_drop - rows_after_drop} inconsistent rows.")

# Apply aggressive cleaning to the training data itself
df_combined['text'] = df_combined['text'].apply(clean_input_text)
df_combined['label'] = df_combined['label'].astype(int)

# --- 4. Prepare Data for Training (Full Set) ---
X = df_combined['text'].fillna('').astype(str) 
y = df_combined['label']

# Split Data (for accuracy report ONLY)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# --- 5. INITIAL Training (for Accuracy Report) ---
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
tfidf_test = tfidf_vectorizer.transform(X_test)
pac = PassiveAggressiveClassifier(max_iter=50) 
pac.fit(tfidf_train, y_train)

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"Initial Model Accuracy on Test Data: {score:.4f}")


# --- 6. FINAL TRAINING ON 100% OF DATA (For Deployment) ---
print("\n--- Final Training on 100% of Cleaned Data ---")

# Re-fit vectorizer on FULL, CLEANED dataset (X)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_full = tfidf_vectorizer.fit_transform(X)

# Final model trained on 100% of the data
pac_final = PassiveAggressiveClassifier(max_iter=50) 
pac_final.fit(tfidf_full, y)

# --- 7. Save the FITTED Vectorizer and the FINAL Trained Model ---
joblib.dump(tfidf_vectorizer, 'tfidfvect.pkl')
joblib.dump(pac_final, 'model.pkl')

print("\n--- SUCCESSFULLY saved UPDATED 'tfidfvect.pkl' and 'model.pkl' ---")
print("The model is now trained on a robust dataset and is ready for deployment.")