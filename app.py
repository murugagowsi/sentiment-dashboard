import joblib
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---- Page Configuration ----
st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="wide"
)

st.title("Product Review Sentiment Dashboard")


# ---- Dataset URL ----
CSV_URL = (
    "https://raw.githubusercontent.com/"
    "murugagowsi/sentiment-dashboard/"
    "refs/heads/main/amazon_reviews_sample.csv"
)


# ---- Clean Text ----
def clean_text(text):
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower()


# ---- Custom Stopwords ----
custom_stopwords = STOPWORDS.union({
    "br", "product", "one", "will", "amazon", "much", "even",
    "really", "got", "get", "just", "also", "would", "like",
    "im", "ive", "dont", "didnt", "cant", "thing", "bit", "lot",
    "now", "say", "says", "time", "buy", "bought", "order",
    "ordered", "use", "used", "make", "made", "know", "think",
    "thought", "try", "tried", "found", "came", "box", "bag",
    "bags"
})


# ---- Load Dataset ----
uploaded = st.file_uploader(
    "Upload your own CSV (optional)",
    type=["csv"]
)


@st.cache_data
def load_default():
    return pd.read_csv(CSV_URL)


try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = load_default()

    st.success(f"Loaded {len(df)} reviews!")

except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()


# ---- Find Review Text Column ----
text_col = next(
    (
        c for c in df.columns
        if c.lower() in [
            "text",
            "review",
            "reviews",
            "review_text",
            "clean_text"
        ]
    ),
    None
)

if text_col is None:
    st.error(
        f"No review text column found. "
        f"Available columns: {list(df.columns)}"
    )
    st.stop()


# ---- Find Sentiment Column ----
valid_sentiments = {"Positive", "Neutral", "Negative"}

df["sent"] = None

for column in ["sentiment_label", "sentiment", "label"]:
    if column in df.columns:
        values = df[column].astype(str).str.capitalize()

        if set(values.unique()).issubset(valid_sentiments):
            df["sent"] = values
            break


# ---- Derive Sentiment from Score if Needed ----
if df["sent"].isna().all():

    if "Score" in df.columns:
        df["sent"] = df["Score"].apply(
            lambda score:
                "Positive" if score >= 4
                else "Neutral" if score == 3
                else "Negative"
        )

    else:
        st.error(
            "No usable sentiment column found. "
            f"Available columns: {list(df.columns)}"
        )
        st.stop()


# ---- Clean Review Text for Word Cloud ----
df["wc_text"] = df[text_col].apply(clean_text)


# ---- Dataset Preview ----
st.subheader("Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)


# ---- Sentiment Distribution ----
st.subheader("Sentiment Distribution")

counts = df["sent"].value_counts()

colors = {
    "Positive": "#2ecc71",
    "Neutral": "#f1c40f",
    "Negative": "#e74c3c"
}

if not counts.empty:

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.pie(
        counts.values,
        labels=None,
        autopct="%1.1f%%",
        colors=[
            colors.get(label, "#95a5a6")
            for label in counts.index
        ],
        startangle=90
    )

    ax.legend(
        counts.index,
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )

    ax.axis("equal")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.pyplot(fig)

    plt.close(fig)

    st.caption(
        "Note: The dataset may contain more positive reviews "
        "than negative reviews."
    )


# ---- Word Cloud for Negative Reviews ----
st.subheader("Top Words in Negative Reviews")

negative_reviews = df.loc[
    df["sent"] == "Negative",
    "wc_text"
]

negative_text = " ".join(negative_reviews.dropna().astype(str))

if negative_text.strip():

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=custom_stopwords
    ).generate(negative_text)

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    ax2.axis("off")

    st.pyplot(fig2)

    plt.close(fig2)

else:
    st.info("No negative reviews found in this dataset.")
# ---- Model vs VADER ----
st.subheader("Model vs VADER")

st.table(pd.DataFrame({
    "Metric": ["Macro-F1", "Negative recall", "Accuracy"],
    "VADER": ["0.49", "43%", "~78%"],
    "My Model": ["0.53", "63%", "~78%"],
}))

# ---- Try It Yourself ----
st.subheader("Try It Yourself")

st.write(
    "Enter a product review below to analyze its sentiment."
)


# ---- Load Machine Learning Model ----
@st.cache_resource
def load_model():

    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    return model, vectorizer


# ---- Review Input ----
user_review = st.text_area(
    "Type a product review:",
    placeholder="Example: This product is excellent and works perfectly!"
)


# ---- Analyze Review ----
if st.button("Analyze"):

    if not user_review.strip():

        st.warning("Please type a review first.")

    else:

        try:
            # Machine Learning Prediction
            ml_model, ml_vectorizer = load_model()

            cleaned_review = clean_text(user_review)

            review_vector = ml_vectorizer.transform(
                [cleaned_review]
            )

            probs = ml_model.predict_proba(review_vector)[0]
            prediction = ml_model.predict(review_vector)[0]
            confidence = probs.max() * 100

            # Low confidence means the review is treated as Neutral
            
            low_confidence=confidence<60

            # Display ML prediction
            st.write(
                f"**My Model:** {str(prediction).capitalize()}"
            )

            # Display confidence
            st.write(
                f"**Model Confidence:** {confidence:.0f}%"
            )
            st.write(dict(zip(ml_model.classes_, probs.round(2))))

            # VADER Sentiment Analysis
            vader_analyzer = SentimentIntensityAnalyzer()

            vader_score = vader_analyzer.polarity_scores(
                user_review
            )["compound"]

            if vader_score >= 0.05:
                vader_label = "Positive"

            elif vader_score <= -0.05:
                vader_label = "Negative"

            else:
                vader_label = "Neutral"

            st.write(
                f"**VADER:** {vader_label} "
                f"(Score: {vader_score:.2f})"
            )

        except FileNotFoundError:
            st.error(
                "Model files not found. Please ensure "
                "model.pkl and vectorizer.pkl exist "
                "in your GitHub repository."
            )

        except Exception as e:
            st.error(f"Could not analyze the review: {e}")
