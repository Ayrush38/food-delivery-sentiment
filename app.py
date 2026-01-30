import pandas as pd
import re
import os
import nltk
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# NLTK setup
# ===============================
NLTK_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", download_dir=NLTK_DIR)

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("food_delivery_reviews.csv")

# ===============================
# Text Preprocessing
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

df["clean_review"] = df["review"].apply(clean_text)

# ===============================
# VADER Sentiment Labeling
# ===============================
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["clean_review"].apply(get_sentiment)

# ===============================
# ML Model
# ===============================
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["clean_review"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(
    y_test, y_pred,
    labels=["Negative", "Neutral", "Positive"]
)

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Food Delivery Sentiment Analyzer")

st.title("🍔 Online Food Delivery Review Sentiment Analyzer")

st.markdown("""
This system analyzes reviews using **VADER Sentiment Analysis**  
and visualizes **sentiment scores for user-input reviews**.
""")

# ===============================
# Dataset Graph
# ===============================
st.subheader("📊 Sentiment Distribution (Dataset)")
fig1, ax1 = plt.subplots()
sns.countplot(x="sentiment", data=df, ax=ax1)
st.pyplot(fig1)

# ===============================
# Model Performance
# ===============================
st.subheader("📈 Model Performance")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.text(classification_report(y_test, y_pred))

# ===============================
# User Input
# ===============================
st.subheader("✍️ Analyze Your Own Review")
user_input = st.text_area("Enter a food delivery review:")

if st.button("Analyze Review"):
    cleaned = clean_text(user_input)

    # ML Prediction
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    # VADER Scores
    scores = sia.polarity_scores(cleaned)

    st.success(f"Predicted Sentiment: **{prediction}**")

    # ===============================
    # GRAPH FOR USER REVIEW
    # ===============================
    st.subheader("📊 Sentiment Score for Your Review")

    score_df = pd.DataFrame({
        "Sentiment": ["Negative", "Neutral", "Positive"],
        "Score": [scores["neg"], scores["neu"], scores["pos"]]
    })

    fig2, ax2 = plt.subplots()
    sns.barplot(x="Sentiment", y="Score", data=score_df, ax=ax2)
    ax2.set_ylim(0, 1)
    st.pyplot(fig2)

    st.write("**VADER Compound Score:**", scores["compound"])
