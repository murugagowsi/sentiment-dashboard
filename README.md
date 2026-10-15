

## Related Project

A separate FastAPI backend for the same model and VADER analysis is available here: [sentiment-api](https://github.com/murugagowsi/sentiment-api). It returns both a trained ML prediction and a VADER prediction for any review.
# Product Review Sentiment Dashboard

An interactive Streamlit dashboard that analyzes Amazon food product reviews and shows sentiment trends and common complaints.

**Live app:** https://sentiment-dashboard-a5xewubpuz6xabmbwuqsto.streamlit.app/

![Data preview](Screenshot%202026-10-11%20225807.png)


<img src="Screenshot%202026-10-11%20231515.png" width="500">




![Negative review word cloud](Screenshot%202026-10-11%20230313.png)



## Features
- Auto-loads 2,113 Amazon food reviews
- Optional upload of your own CSV
- Sentiment distribution pie chart
- Word cloud of top words in negative reviews
- "Try it yourself" box that scores any review with VADER sentiment

## Tools Used
- Python
- pandas
- matplotlib
- WordCloud
- Streamlit
- VADER (vaderSentiment)

## Key Insights
- **Imbalanced data:** 77.6% positive, 14.2% negative, 8.2% neutral
- **Quality is the top complaint:** "taste", "flavor", "salty", "hard", "bad"
- **Chips are the most complained-about category**, followed by coffee, cookies and frosting
- **Value and packaging also come up:** "price", "money", "package", "plastic"
- **Limitation:** words like "good" and "great" appear in negative reviews, likely from phrases like "not good". The "Try it yourself" box handles these cases with VADER.
## Data Cleaning
- Removed HTML tags (e.g. `<br />`)
- Kept letters only and lowercased text
- Added a custom stopword list for generic words

## What I Learned
- Cleaning messy text data before visualizing it
- Building and deploying a Streamlit app
- Reading word clouds critically, since frequency does not equal meaning

## Future Improvements
- Use bigrams (e.g. "not good", "too salty") 
- Balance the dataset before modeling
  
