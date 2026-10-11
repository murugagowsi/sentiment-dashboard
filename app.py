import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("📊 Amazon Review Sentiment Analyzer")
st.markdown("---")

@st.cache_data
def load_data_from_github():
    """Load sample data from GitHub"""
    url = "https://raw.githubusercontent.com/murugagowski/sentiment-analysis/main/amazon_reviews_sample.csv"
    return pd.read_csv(url, engine='python', on_bad_lines='skip')

@st.cache_data
def load_data_from_upload(file):
    """Load user-uploaded data"""
    return pd.read_csv(file, engine='python', on_bad_lines='skip')

# Try to load GitHub data first, then allow upload
try:
    df = load_data_from_github()
    st.success(f"✅ Loaded {len(df)} reviews from sample dataset!")
except:
    st.info("📤 Upload your CSV file to analyze")
    df = None

uploaded_file = st.file_uploader("Or upload your own CSV file", type="csv")

if uploaded_file:
    df = load_data_from_upload(uploaded_file)
    st.success(f"✅ Loaded {len(df)} reviews from your file!")

if df is not None:
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.subheader("📈 Sentiment Distribution")
    col1, col2, col3 = st.columns(3)
    positive = (df["sentiment_label"] == 2.0).sum()
    neutral = (df["sentiment_label"] == 1.0).sum()
    negative = (df["sentiment_label"] == 0.0).sum()
    with col1:
        st.metric("✅ Positive", positive)
    with col2:
        st.metric("⚪ Neutral", neutral)
    with col3:
        st.metric("❌ Negative", negative)
    st.subheader("Sentiment Pie Chart")
    fig, ax = plt.subplots(figsize=(8, 6))
    sentiment_counts = df["sentiment_label"].value_counts()
    sentiment_labels = ["Positive" if x == 2.0 else "Neutral" if x == 1.0 else "Negative" for x in sentiment_counts.index]
    ax.pie(sentiment_counts.values, labels=sentiment_labels, autopct="%1.1f%%", colors=['#2ecc71', '#f39c12', '#e74c3c'])
    st.pyplot(fig)
    st.subheader("🔤 Top Words in Negative Reviews")
    negative_reviews = df[df["sentiment_label"] == 0.0]["clean_text"].tolist()
    if len(negative_reviews) > 0:
        negative_text = " ".join(negative_reviews)
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    st.subheader("📋 Top Complaint Topics")
    words = []
    stop_words = {'the', 'a', 'is', 'it', 'this', 'that', 'and', 'or', 'not', 'i', 'you', 'me', 'to', 'in', 'of', 'for', 'was', 'are'}
    for review in negative_reviews:
        review_words = review.lower().split()
        for word in review_words:
            clean_word = re.sub(r'[^a-z0-9]', '', word)
            if len(clean_word) > 3 and clean_word not in stop_words:
                words.append(clean_word)
    if words:
        top_words = Counter(words).most_common(15)
        for word, count in top_words:
            st.write(f"• {word}: {count}")
else:
    st.info("👆 Upload a CSV file to get started!")
