# ==============================
# 1. IMPORT LIBRARIES
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stopwords (runs only first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# ==============================
# 2. LOAD MODEL & TOKENIZER
# ==============================

# 🔥 IMPORTANT: Ensure these files exist in this path
model = tf.keras.models.load_model(r"C:\Users\KAVIYA V\best_model.keras")
tokenizer = pickle.load(open(r"C:\Users\KAVIYA V\tokenizer.pkl", "rb"))

MAX_LEN = 150

label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']


# ==============================
# 3. TEXT CLEANING FUNCTION
# ==============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    return " ".join(words)


# ==============================
# 4. PREDICTION FUNCTION
# ==============================

def predict_toxicity(text):
    text = clean_text(text)
    
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict(padded)[0]

    result = {}
    toxic_flag = False

    for label, prob in zip(label_cols, pred):
        prob = float(prob)

        if prob > 0.3:
            toxic_flag = True

        result[label] = {
            "probability": round(prob, 3),
            "label": "Toxic" if prob > 0.3 else "Non-Toxic"
        }

    result["overall"] = "🔴 Toxic Comment" if toxic_flag else "🟢 Safe Comment"
    
    return result


# ==============================
# 5. STREAMLIT UI SETUP
# ==============================

st.set_page_config(page_title="Toxicity Detection", layout="wide")

st.title("💬 Comment Toxicity Detection App")
st.markdown("Deep Learning model to detect toxic comments in real-time")


# ==============================
# 6. SIDEBAR NAVIGATION
# ==============================

menu = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Data Insights", "Prediction", "About"]
)


# ==============================
# 7. HOME PAGE
# ==============================

if menu == "Home":
    st.header("🏠 Project Overview")
    
    st.write("""
    This application detects toxic comments using Deep Learning models (CNN & LSTM).

    🔹 Features:
    - Real-time prediction
    - Multi-label toxicity detection
    - Clean and interactive UI
    """)


# ==============================
# 8. DATA INSIGHTS
# ==============================

elif menu == "Data Insights":
    st.header("📊 Data Insights")

    df = pd.read_csv(r"C:\Users\KAVIYA V\Downloads\train.csv")

    label_sum = df[label_cols].sum()

    fig, ax = plt.subplots(figsize=(5,3))
    sns.barplot(x=label_cols, y=label_sum.values, ax=ax)
    plt.xticks(rotation=45)
    plt.title("Label Distribution")

    st.pyplot(fig, use_container_width=True)


# ==============================
# 9. SINGLE PREDICTION
# ==============================

elif menu == "Prediction":
    st.header("🧠 Predict Toxicity")

    user_input = st.text_area("Enter a comment:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter text")
        else:
            result = predict_toxicity(user_input)

            st.subheader("Result:")
            st.write(result["overall"])

            # Convert result to table
            df_result = pd.DataFrame(result).T
            df_result = df_result.drop("overall", errors='ignore')

            st.dataframe(df_result)


# ==============================
# 10. ABOUT PAGE ⭐
# ==============================

elif menu == "About":
    st.header("📌 Project Summary")

    st.markdown("""
    ### 💬 Comment Toxicity Detection System

    This project uses Deep Learning models (CNN & LSTM) to identify toxic comments in real-time.

    ###  Key Highlights:
    - Multi-label classification (6 toxicity types)
    - Real-time prediction system
    - Model comparison (CNN vs LSTM)
    - Clean UI using Streamlit

    ###  Technologies Used:
    - Python
    - TensorFlow / Keras
    - NLP (Text preprocessing)
    - Streamlit """)

    st.markdown("""
### 🚀 Final Output:
The system predicts whether a comment is **Toxic or Safe**, along with probabilities for each category. 
\n✅ Project Completed Successfully! """)

