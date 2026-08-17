# Bahrain Restaurant Review Analyzer

## Overview

This project analyzes restaurant reviews using pretrained Hugging Face models.

The project focuses on restaurant reviews collected from Bahrain and explores how different pretrained sentiment analysis models perform on real-world review data containing both English and Arabic text.

The application also uses zero-shot classification to identify the main topic discussed in a review.

## Features

- Sentiment classification of restaurant reviews
- Positive, neutral, and negative sentiment prediction
- Confidence score for each prediction
- Zero-shot topic classification
- Support for English and Arabic reviews
- Comparison of multiple pretrained Hugging Face models
- Model evaluation using classification metrics
- Interactive Streamlit application

## Sentiment Models

Three pretrained sentiment models were evaluated:

1. `cardiffnlp/twitter-xlm-roberta-base-sentiment`
2. `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`
3. `lxyuan/distilbert-base-multilingual-cased-sentiments-student`

The models were compared using restaurant reviews from Bahrain to examine their performance across different languages and review styles.

## Zero-Shot Classification

Zero-shot classification is used to identify the main topic of a restaurant review without training a new classifier.

The topics used are:

- Food Quality
- Service
- Price
- Cleanliness
- Atmosphere
- Location
- Waiting Time

The zero-shot model used is:

`facebook/bart-large-mnli`

## Dataset

The dataset was collected from publicly available Google Maps reviews for Zahlawiya Cafeteria in Bahrain.

The reviews were extracted using Apify's Google Maps Reviews Scraper and exported as a CSV file.

The main fields used in the analysis are:

- Review text
- Star rating

Reviews without written text were removed before analysis.

Star ratings were converted into sentiment labels:

- 4–5 stars: Positive
- 3 stars: Neutral
- 1–2 stars: Negative

The collected data was used only for model evaluation and comparison.

## Evaluation

The sentiment models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Error Analysis

Incorrect predictions were also reviewed to understand common failure cases and differences between the models.

## Application

The Streamlit application allows a user to enter a restaurant review and receive:

- Predicted sentiment
- Sentiment confidence score
- Main review topic
- Topic confidence score

## Technologies Used

- Python
- Hugging Face Transformers
- PyTorch
- Pandas
- Scikit-learn
- Streamlit
- Google Colab

## Running the Application

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

```text
bahrain-restaurant-review-analyzer/
├── app.py
├── requirements.txt
├── README.md
├── lastex_completed.ipynb
└── zahlawiya_clean.csv
```
