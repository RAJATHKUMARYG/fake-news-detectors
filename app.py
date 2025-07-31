import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime
import csv

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page config
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🕵️", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .main-container {
            background-color: #f5f7fa;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            font-family: 'Segoe UI', sans-serif;
        }
        .admin-button {
            position: absolute;
            top: 20px;
            left: 20px;
            background-color: #4B9CD3;
            color: white;
            padding: 10px 18px;
            border-radius: 8px;
            font-weight: bold;
            text-decoration: none;
        }
        .result-real {
            background-color: #d4edda;
            color: #155724;
            padding: 12px;
            border-radius: 8px;
            font-weight: bold;
        }
        .result-fake {
            background-color: #f8d7da;
            color: #721c24;
            padding: 12px;
            border-radius: 8px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Admin button
if st.button("🛠️ Admin Panel (View History)"):
    if os.path.exists("prediction_logs.csv"):
        logs_df = pd.read_csv("prediction_logs.csv", header=None)
        logs_df.columns = ["Timestamp", "News Snippet", "Prediction"]
        st.markdown("### 🧾 Prediction History")
        st.dataframe(logs_df, use_container_width=True)

        csv_download = logs_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Logs", data=csv_download,
                           file_name="prediction_logs.csv", mime="text/csv")
    else:
        st.warning("No history found yet.")

# App heading
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("## 📰 Fake News Detector")
st.write("Paste your news content below and check whether it's **Real** or **Fake** using Machine Learning.")

# User input
user_input = st.text_area("✏️ Enter News Content Here:", height=200)

# Predict
if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter some news text.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)
        result = "REAL" if prediction[0] == 1 else "FAKE"

        # Show result
        if result == "REAL":
            st.markdown('<div class="result-real">✅ This news appears to be REAL.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-fake">🚫 This news appears to be FAKE.</div>', unsafe_allow_html=True)

        # Save to log
        with open("prediction_logs.csv", mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now(), user_input[:100], result])

st.markdown("</div>", unsafe_allow_html=True)
