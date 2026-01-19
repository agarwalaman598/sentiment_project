# 🎭 Real-Time Sentiment Analysis Dashboard

A comprehensive sentiment analysis project that combines traditional Machine Learning (Logistic Regression + TF-IDF) with modern deep learning (BERT) to analyze social media text sentiment in real-time.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-Latest-yellow.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)

---

## ✨ Features

- **Dual Model Architecture**: Compare predictions from both ML (Logistic Regression) and BERT models
- **Real-Time Analysis**: Interactive Streamlit dashboard for instant sentiment predictions
- **Twitter Scraping**: Built-in scraper to collect social media data for training
- **Data Pipeline**: Complete preprocessing pipeline with text cleaning and normalization
- **Confidence Scores**: BERT model provides confidence scores for predictions

---

## 📁 Project Structure

```
sentiment_project/
├── app.py                    # Streamlit dashboard application
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
├── .gitignore                # Git ignore rules
│
├── data/
│   ├── raw/                  # Raw scraped data
│   │   └── tweets.csv
│   └── processed/            # Cleaned and processed data
│       └── cleaned.csv
│
├── models/
│   ├── bert_model/           # Fine-tuned BERT model (if applicable)
│   └── ml_model.pkl          # Trained ML model + vectorizer
│
├── notebooks/                # Jupyter notebooks for experimentation
│
└── src/
    ├── scraper.py            # Twitter/social media scraper
    ├── preprocess.py         # Data cleaning and preprocessing
    ├── train_ml.py           # ML model training (Logistic Regression)
    ├── train_bert.py         # BERT model usage/fine-tuning
    └── predict.py            # Prediction module with CLI
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sentiment_project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (required for preprocessing)
   ```python
   import nltk
   nltk.download('stopwords')
   ```

---

## 📦 Dependencies

Add these to your `requirements.txt`:

```
streamlit
pandas
scikit-learn
transformers
torch
joblib
nltk
snscrape
```

---

## 🔧 Usage

### 1. Data Collection (Optional)

Scrape tweets for a specific keyword:

```bash
python src/scraper.py
```

> **Note**: Modify the keyword and limit in `scraper.py` as needed.

### 2. Data Preprocessing

Clean and preprocess the raw data:

```bash
python src/preprocess.py
```

This will:
- Remove URLs, mentions, and hashtags
- Convert text to lowercase
- Remove special characters
- Remove stopwords
- Save cleaned data to `data/processed/cleaned.csv`

### 3. Train ML Model

Train the Logistic Regression model:

```bash
python src/train_ml.py
```

This will:
- Load the preprocessed data
- Train a TF-IDF + Logistic Regression pipeline
- Save the model to `models/ml_model.pkl`
- Print classification metrics

### 4. Run the Dashboard

Launch the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 5. CLI Prediction (Alternative)

For command-line predictions:

```bash
python src/predict.py
```

---

## 🎯 Model Details

### ML Model (Logistic Regression)
- **Vectorizer**: TF-IDF with 5,000 max features
- **Classifier**: Logistic Regression
- **Labels**: Negative, Neutral, Positive

### BERT Model
- **Model**: HuggingFace `sentiment-analysis` pipeline
- **Pre-trained**: distilbert-base-uncased-finetuned-sst-2-english
- **Output**: Label + Confidence Score

---

## 📊 Data Format

### Input Data (`data/raw/tweets.csv`)
| Column | Description |
|--------|-------------|
| date | Tweet timestamp |
| text | Tweet content |

### Processed Data (`data/processed/cleaned.csv`)
| Column | Description |
|--------|-------------|
| sentiment | Sentiment label (0=negative, 2=neutral, 4=positive) |
| text | Original text |
| clean_text | Preprocessed text |

---

## 🖥️ Dashboard Preview

The Streamlit dashboard provides:
- Text input area for entering social media text
- Analyze button to trigger predictions
- Side-by-side comparison of ML and BERT predictions
- Confidence scores for BERT predictions

---

## 🛠️ Development

### Adding New Models

1. Create a new training script in `src/`
2. Save the trained model to `models/`
3. Update `predict.py` and `app.py` to load the new model

### Extending the Scraper

Modify `src/scraper.py` to:
- Change search keywords
- Adjust the number of tweets to scrape
- Add additional metadata fields

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

<p align="center">
  Made with ❤️ for sentiment analysis
</p>
