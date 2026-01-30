import pandas as pd
import re
import nltk
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download VADER lexicon
nltk.download("vader_lexicon")

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("food_delivery_reviews.csv")

# ===============================
# Text Preprocessing
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = text.strip()
    return text

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
# Feature Extraction (TF-IDF)
# ===============================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_review"])
y = df["sentiment"]

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Naive Bayes Classification
# ===============================
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral", "Positive"])

# ===============================
# STREAMLIT UI
# ===============================
st.title("🍔 Online Food Delivery Review Sentiment Analyzer")

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Sentiment Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x="sentiment", data=df, ax=ax1)
st.pyplot(fig1)

st.subheader("Model Performance")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
fig2, ax2 = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Neutral", "Positive"],
    yticklabels=["Negative", "Neutral", "Positive"]
)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

st.subheader("Try Your Own Review")
user_input = st.text_area("Enter a food delivery review:")

if st.button("Analyze Sentiment"):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    st.success(f"Predicted Sentiment: **{prediction}**")
