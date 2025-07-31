import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime
import csv
import json
from streamlit_lottie import st_lottie

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load Lottie animation
def load_lottie_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

lottie_animation = load_lottie_file("animation.json")  # add any animation file you like

# --- Page Config ---
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
        .main-box {
            background-color: #e0f0ff;
            padding: 2rem 2rem 2rem 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
            font-family: 'Segoe UI', sans-serif;
            margin-top: 1rem;
        }
        .result-real {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 10px;
            font-weight: bold;
        }
        .result-fake {
            background-color: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 10px;
            font-weight: bold;
        }
        .admin-box {
            position: absolute;
            top: 20px;
            left: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Admin Auth State ---
if "admin_mode" not in st.session_state:
    st.session_state.admin_mode = False

# --- Admin Login Button ---
with st.sidebar:
    st.subheader("🔐 Admin Access")
    if not st.session_state.admin_mode:
        password = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if password == "admin123":  # change this to a more secure password
                st.session_state.admin_mode = True
                st.success("Access granted!")
            else:
                st.error("Wrong password")
    else:
        st.success("Admin Logged In")
        if st.button("Logout"):
            st.session_state.admin_mode = False

# --- Main Layout ---
with st.container():
    st_lottie(lottie_animation, height=250)
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    st.title("📰 Fake News Detector")
    st.write("Paste news content below and detect whether it's REAL or FAKE.")

    news_input = st.text_area("Enter News Article", height=180)

    if st.button("🔍 Predict"):
        if not news_input.strip():
            st.warning("Please enter some news content.")
        else:
            vec = vectorizer.transform([news_input])
            pred = model.predict(vec)
            result = "REAL" if pred[0] == 1 else "FAKE"

            # Display Result
            if result == "REAL":
                st.markdown('<div class="result-real">✅ This news appears to be REAL.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-fake">🚫 This news appears to be FAKE.</div>', unsafe_allow_html=True)

            # Log result
            with open("prediction_logs.csv", mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), news_input[:100], result])

    st.markdown("</div>", unsafe_allow_html=True)

# --- Admin Panel ---
if st.session_state.admin_mode:
    st.markdown("## 🛠️ Admin Panel - Prediction History")
    if os.path.exists("prediction_logs.csv"):
        df = pd.read_csv("prediction_logs.csv", header=None)
        df.columns = ["Timestamp", "News Snippet", "Prediction"]
        st.dataframe(df, use_container_width=True)

        # Download CSV
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Logs", data=csv_data, file_name="prediction_logs.csv", mime="text/csv")
    else:
        st.info("No prediction history yet.")
