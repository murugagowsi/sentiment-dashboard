import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("Product Review Sentiment Dashboard")

# ---------- CHANGE THIS: your CSV file name in GitHub ----------
CSV_URL = "https://raw.githubusercontent.com/murugagowski/sentiment-analysis/main/sample_reviews.csv"

# ---------- Cleaning ----------
def clean_text(text):
    text = re.sub(r"<.*?>", " ", str(text))   # remove HTML tags like <br />
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # letters only
    return text.lower()

custom_stopwords = STOPWORDS.union({
    "br", "product", "one", "will", "amazon", "much", "even",
    "really", "got", "get", "just", "also", "would", "like",
    "im", "ive", "dont", "didnt", "cant", "thing", "bit", "lot"
})

# ---------- Load data ----------
uploaded = st.file_uploader("Upload your own CSV (optional)", type="csv")

@st.cache_data
def load_default():
    return pd.read_csv(CSV_URL)

try:
    df = pd.read_csv(uploaded) if uploaded else load_default()
    st.success(f"Loaded {len(df)} reviews!")
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# ---------- Detect columns ----------
text_col = next((c for c in df.columns if c.lower() in ["text", "review", "reviews", "review_text"]), None)
if text_col is None:
    st.error(f"No review text column found. Columns: {list(df.columns)}")
    st.stop()

sent_col = next((c for c in df.columns if c.lower() in ["sentiment", "label"]), None)
if sent_col is None:
    score_col = next((c for c in df.columns if c.lower() in ["score", "rating"]), None)
    if score_col is None:
        st.error(f"No sentiment or score column found. Columns: {list(df.columns)}")
        st.stop()
    df["sentiment"] = df[score_col].apply(
        lambda s: "Positive" if s >= 4 else ("Neutral" if s == 3 else "Negative")
    )
    sent_col = "sentiment"

df[sent_col] = df[sent_col].astype(str).str.capitalize()
df["clean_text"] = df[text_col].apply(clean_text)

# ---------- Preview ----------
st.subheader("Dataset Preview")
st.dataframe(df.head(10))

# ---------- Pie chart ----------
st.subheader("Sentiment Distribution")
counts = df[sent_col].value_counts()
colors = {"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"}

fig, ax = plt.subplots(figsize=(4, 4))
ax.pie(
    counts.values,
    autopct="%1.1f%%",
    colors=[colors.get(k, "#95a5a6") for k in counts.index],
    startangle=90,
)
ax.legend(counts.index, loc="center left", bbox_to_anchor=(1, 0.5))
ax.axis("equal")
st.pyplot(fig, use_container_width=False)

st.caption("Note: the dataset is imbalanced, with far more positive reviews than negative ones.")

# ---------- Word cloud ----------
st.subheader("Top Words in Negative Reviews")
neg_text = " ".join(df[df[sent_col] == "Negative"]["clean_text"])

if neg_text.strip():
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        stopwords=custom_stopwords,
    ).generate(neg_text)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.imshow(wc, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)
else:
    st.info("No negative reviews found.")
