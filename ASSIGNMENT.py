# Import necessary libraries
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Sample dataset of product reviews
data = {
    "Product": ["Phone", "Laptop", "Headphones", "Smartwatch", "Camera"],
    "Reviews": [
        "Amazing phone with great battery life! I love it.",
        "Terrible laptop. Slow performance and battery drains fast.",
        "Headphones have excellent sound quality but are overpriced.",
        "Smartwatch is just okay. Nothing special, but it works fine.",
        "Camera takes stunning photos! Worth every penny."
    ]
}

# Convert dataset to DataFrame
df = pd.DataFrame(data)

# Initialize NLTK's stopwords and Sentiment Analyzer
stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

# Function to clean and preprocess reviews
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    words = text.split()  # Tokenize
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Apply preprocessing to the reviews
df["Processed_Reviews"] = df["Reviews"].apply(preprocess_text)

# Function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(text):
    sentiment = TextBlob(text).sentiment.polarity  # Polarity ranges from -1 to 1
    return sentiment

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    sentiment = sia.polarity_scores(text)["compound"]  # Compound score ranges from -1 to 1
    return sentiment

# Apply sentiment analysis
df["Sentiment_TextBlob"] = df["Processed_Reviews"].apply(analyze_sentiment_textblob)
df["Sentiment_VADER"] = df["Processed_Reviews"].apply(analyze_sentiment_vader)

# Function to classify sentiment
def classify_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply classification based on VADER sentiment scores
df["Sentiment_Label"] = df["Sentiment_VADER"].apply(classify_sentiment)

# Display the DataFrame
print(df[["Product", "Reviews", "Sentiment_Label"]])

# Visualization - Sentiment distribution
sentiment_counts = df["Sentiment_Label"].value_counts()
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind="bar", color=["green", "red", "blue"])
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Distribution of Product Reviews")
plt.xticks(rotation=0)
plt.show()
