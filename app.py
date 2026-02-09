# =========================================
# STREAMLIT SENTIMENT ANALYSIS APP
# =========================================

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once (safe for Streamlit)
nltk.download("stopwords")
nltk.download("wordnet")

# =========================================
# LOAD MODEL & VECTORIZER
# =========================================
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# =========================================
# TEXT PREPROCESSING (SAME AS TRAINING)
# =========================================
stop_words = set(stopwords.words("english")) - {"not", "no", "never"}
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# =========================================
# STREAMLIT UI CONFIG
# =========================================
st.set_page_config(page_title="Flipkart Sentiment Analysis", layout="centered")

# Custom CSS Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2E86C1;
        text-align: center;
    }
    .stTextArea textarea {
        border: 2px solid #2E86C1;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================
# HEADER & LOGO
# =========================================

st.title("🛒 Flipkart Review Sentiment Analysis")
st.write("Enter a product review and predict its sentiment")

# =========================================
# INPUT AREA
# =========================================
review_text = st.text_area(
    "Enter your review here:",
    height=150,
    placeholder="Example: This product quality is amazing and worth the price!"
)

# =========================================
# PREDICTION
# =========================================
if st.button("🔍 Predict Sentiment"):
    if review_text.strip() == "":
        st.warning("⚠️ Please enter a review text.")
    else:
        clean_text = preprocess(review_text)
        vectorized_text = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.success("🎉 Positive Review – Customers love it!")
        else:
            st.error("⚠️ Negative Review – Needs improvement.")

# =========================================
# SIDEBAR INFO
# =========================================
st.sidebar.title("ℹ️ About")
st.sidebar.write("This app analyzes Flipkart product reviews using **TF-IDF + Logistic Regression**.")
st.sidebar.write("Built with Streamlit 💻")

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)