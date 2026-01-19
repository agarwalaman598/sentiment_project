# Main application entry point
import streamlit as st
import pandas as pd
import joblib
from transformers import pipeline

st.set_page_config(page_title="Sentiment Dashboard", layout="centered")
st.title("Real-Time Sentiment Analysis Dashboard")

@st.cache_resource
def load_models():
    vectorizer, model = joblib.load("models/ml_model.pkl")
    bert = pipeline("sentiment-analysis")
    return vectorizer, model, bert

st.write("Loading models...")
vectorizer, model, bert = load_models()
st.success("Models loaded")

text_input = st.text_area("Enter social media text")

if st.button("Analyze"):
    ml_pred = model.predict(vectorizer.transform([text_input]))[0]
    bert_pred = bert(text_input)[0]

    st.subheader("Results")
    st.write("ML Model Prediction:", ml_pred)
    st.write("BERT Prediction:", bert_pred["label"], f"(Confidence: {bert_pred['score']:.2f})")
