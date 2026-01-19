import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

df = pd.read_csv("data/raw/tweets.csv", header=None, encoding="latin-1")

df = df[[0, 5]]
df.columns = ["sentiment", "text"]

def clean(text):
    text = re.sub(r"http\S+|@\S+|#\S+", "", str(text).lower())
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(w for w in text.split() if w not in stop_words)

df["clean_text"] = df["text"].apply(clean)

df.to_csv("data/processed/cleaned.csv", index=False)

print("Saved: data/processed/cleaned.csv")
