import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("Product Review Sentiment Dashboard")

# CHANGE THIS if your CSV file name in GitHub is different
CSV_URL = "https://raw.githubusercontent.com/murugagowski/sentiment-analysis/main/sample_reviews.csv"

def clean_text(text):
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower()

custom_stopwords = STOPWORDS.union({
    "br", "product", "one", "will", "amazon", "much", "even",
    "really", "got", "get", "just", "also", "would", "like",
    "im", "ive", "dont", "didnt", "cant", "thing", "bit", "lot",
    "now", "say", "says", "time", "buy", "bought", "order", "ordered",
    "use", "used", "make", "made", "know", "think", "thought",
    "try", "tried", "found", "came", "box", "bag", "bags"
})

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

# ---- Find text column (your file has clean_text) ----
text_col = next((c for c in df.columns if c.lower() in
                 ["text", "review", "reviews", "review_text", "clean_text"]), None)
if text_col is None:
    st.error(f"No review text column found. Columns: {list(df.columns)}")
    st.stop()

# ---- Find sentiment column ----
valid = {"Positive", "Neutral", "Negative"}
df["sent"] = None
for c in ["sentiment_label", "sentiment", "label"]:
    if c in df.columns:
        vals = df[c].astype(str).str.capitalize()
        if set(vals.unique()) <= valid:
            df["sent"] = vals
            break

if df["sent"].isna().all():
    if "Score" in df.columns:
        df["sent"] = df["Score"].apply(
            lambda s: "Positive" if s >= 4 else ("Neutral" if s == 3 else "Negative"))
    else:
        st.error(f"No usable sentiment column. Columns: {list(df.columns)}")
        st.stop()

df["wc_text"] = df[text_col].apply(clean_text)

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

# ---- Pie chart ----
st.subheader("Sentiment Distribution")
counts = df["sent"].value_counts()
colors = {"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"}

fig, ax = plt.subplots(figsize=(4, 4))
ax.pie(counts.values, autopct="%1.1f%%",
       colors=[colors.get(k, "#95a5a6") for k in counts.index], startangle=90)
ax.legend(counts.index, loc="center left", bbox_to_anchor=(1, 0.5))
ax.axis("equal")
col1, col2 = st.columns([1, 2])
with col1:
    st.pyplot(fig)
st.caption("Note: the dataset is imbalanced, with far more positive reviews than negative ones.")

# ---- Word cloud ----
st.subheader("Top Words in Negative Reviews")
neg_text = " ".join(df[df["sent"] == "Negative"]["wc_text"])

if neg_text.strip():
    wc = WordCloud(width=800, height=400, background_color="white",
                   stopwords=custom_stopwords).generate(neg_text)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.imshow(wc, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)
else:
    st.info("No negative reviews found.")
