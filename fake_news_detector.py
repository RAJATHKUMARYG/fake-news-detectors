import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load Dataset
df = pd.read_csv("news.csv")  # Make sure this contains "text" and "label" columns

# Text Preprocessing Function
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    words = text.split()
    words = [w for w in words if w not in stop_words]  # Remove stopwords
    return ' '.join(words)

df['text'] = df['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Pipeline with TF-IDF and Classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.7)),
    ('clf', PassiveAggressiveClassifier(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
preds = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Save
joblib.dump(pipeline, "fake_news_model.pkl")
