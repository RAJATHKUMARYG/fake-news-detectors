import streamlit as st
import joblib
import csv
import pandas as pd
from datetime import datetime
import os

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# App title
st.title("📰 Fake News Detector")

# Sidebar navigation
page = st.sidebar.selectbox("📂 Choose Page", ["Predict News", "Admin Panel"])

# 🔍 PAGE 1: Prediction Page
if page == "Predict News":
    st.markdown("Enter the news content below to check if it's **FAKE** or **REAL**.")
    user_input = st.text_area("Paste News Article Here:")

    if st.button("Check News"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            transformed_input = vectorizer.transform([user_input])
            prediction = model.predict(transformed_input)
            result = "REAL" if prediction[0] == 1 else "FAKE"

            if result == "REAL":
                st.success("✅ This news is **REAL**.")
            else:
                st.error("🚫 This news is **FAKE**.")

            # Log to CSV
            with open("prediction_logs.csv", mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now(), user_input[:100], result])

# 🛡️ PAGE 2: Admin Panel
elif page == "Admin Panel":
    st.subheader("🔐 Admin Login Required")

    password = st.text_input("Enter Admin Password:", type="password")
    if password == "admin123":  # Change as needed
        st.success("✅ Access Granted")

        # Load logs
        if os.path.exists("prediction_logs.csv"):
            df = pd.read_csv("prediction_logs.csv", header=None)
            df.columns = ["Timestamp", "News (First 100 chars)", "Prediction"]

            st.subheader("📊 Prediction Logs")
            st.dataframe(df, use_container_width=True)

            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Logs as CSV", csv, "prediction_logs.csv", "text/csv")

        else:
            st.info("ℹ️ No logs found yet.")
    elif password != "":
        st.error("❌ Incorrect password. Try again.")
