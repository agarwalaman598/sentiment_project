# Prediction module for sentiment project
import joblib
from transformers import pipeline

# Load models once
vectorizer, model = joblib.load("models/ml_model.pkl")
bert = pipeline("sentiment-analysis")

def predict_text(text):
    """
    Predict sentiment using both ML and BERT models.
    Returns a dictionary with labels and confidence.
    """
    ml_pred = model.predict(vectorizer.transform([text]))[0]
    bert_pred = bert(text)[0]

    return {
        "ml_model": ml_pred,
        "bert_model": bert_pred["label"],
        "confidence": round(bert_pred["score"], 3)
    }

if __name__ == "__main__":
    print("Sentiment Prediction CLI")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("Enter text: ")
        if user_input.lower() == "exit":
            break
        result = predict_text(user_input)
        print(result)
