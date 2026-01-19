# ML training module for sentiment project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("data/processed/cleaned.csv")
df["sentiment"] = df["sentiment"].map({0: "negative", 2: "neutral", 4: "positive"})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

print(classification_report(y_test, model.predict(X_test_vec)))

joblib.dump((vectorizer, model), "models/ml_model.pkl")
