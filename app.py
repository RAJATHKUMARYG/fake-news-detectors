import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import csv
import os

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit page config
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🕵️‍♂️", layout="centered")

# Page style
st.markdown("""
    <style>
    body {
        background-color: #f2f2f2;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .main {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("📰 Fake News Detector")
st.markdown("### Detect whether a news article is **Real or Fake** using Machine Learning.")

# User input
user_input = st.text_area("🖊️ Paste your news content here:", height=200)

# Predict Button
if st.button("🔍 Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)
        result = "REAL" if prediction[0] == 1 else "FAKE"

        # Display result
        if result == "REAL":
            st.success("✅ This news appears to be **REAL**.")
        else:
            st.error("🚫 This news appears to be **FAKE**.")

        # Save to log
        with open("prediction_logs.csv", mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now(), user_input[:100], result])

# Download Logs Button (if logs exist)
if os.path.exists("prediction_logs.csv"):
    logs_df = pd.read_csv("prediction_logs.csv", header=None)
    logs_df.columns = ["Timestamp", "News Snippet", "Prediction"]

    csv_download = logs_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Prediction Logs", data=csv_download,
                       file_name="prediction_logs.csv", mime="text/csv")
