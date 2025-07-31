import streamlit as st
import joblib
import pandas as pd
import csv
import os
from datetime import datetime
from streamlit_lottie import st_lottie
import json

# --- Load Model & Vectorizer ---
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Load Animation ---
def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_animation = load_lottie("animation.json")  # animation file (download from lottiefiles.com)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .blue-box {
        background-color: #e3f2fd;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .centered {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .title-text {
        font-size: 36px;
        font-weight: bold;
        color: #0d47a1;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Admin Login ---
st.sidebar.title("🔒 Admin Access")
admin_password = st.sidebar.text_input("Enter Admin Password", type="password")

# --- Main Page ---
st.markdown('<div class="centered"><div class="title-text">📰 Fake News Detector</div></div>', unsafe_allow_html=True)
st_lottie(lottie_animation, height=300, key="lottie")

# --- Page Mode Selection ---
page = st.selectbox("📂 Select Page", ["Predict News", "Admin Panel"])

# --- Page 1: News Prediction ---
if page == "Predict News":
    st.markdown('<div class="blue-box">', unsafe_allow_html=True)
    st.subheader("🔍 Enter News Content:")
    user_input = st.text_area("Paste news article here", height=200)

    if st.button("Detect"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter news content.")
        else:
            input_transformed = vectorizer.transform([user_input])
            prediction = model.predict(input_transformed)
            result = "REAL" if prediction[0] == 1 else "FAKE"

            if result == "REAL":
                st.success("✅ This news appears to be REAL.")
            else:
                st.error("🚫 This news appears to be FAKE.")

            # Log to CSV
            with open("prediction_logs.csv", mode="a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), user_input[:100], result])

    st.markdown('</div>', unsafe_allow_html=True)

# --- Page 2: Admin Panel ---
elif page == "Admin Panel":
    st.subheader("🛡️ Admin Panel")

    if admin_password == "admin123":
        st.success("Access Granted ✅")

        if os.path.exists("prediction_logs.csv"):
            df = pd.read_csv("prediction_logs.csv", header=None)
            df.columns = ["Timestamp", "News (First 100 chars)", "Prediction"]

            st.dataframe(df, use_container_width=True)

            # Download CSV
            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Logs", data=csv_download, file_name="prediction_logs.csv", mime="text/csv")
        else:
            st.info("No logs found yet.")
    elif admin_password:
        st.error("Incorrect password ❌")
