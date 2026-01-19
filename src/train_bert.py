# BERT training module for sentiment project
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

print(classifier("This project is amazing"))
