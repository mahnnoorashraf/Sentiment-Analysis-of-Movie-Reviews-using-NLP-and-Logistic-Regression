# 📊 Sentiment Analysis of Movie Reviews using NLP and Logistic Regression

This project is a Natural Language Processing (NLP) based machine learning pipeline that performs **sentiment classification** (positive/negative) on movie reviews. It uses data preprocessing, TF-IDF feature extraction, and a Logistic Regression model to predict sentiment with high accuracy.

---

## 🧾 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Preprocessing](#-preprocessing)
- [Model](#-model)
- [Evaluation](#-evaluation)
- [How to Use](#-how-to-use)
- [Requirements](#-requirements)
- [Future Work](#-future-work)

---

## 📌 Overview

The goal is to classify movie reviews into **positive** or **negative** sentiments using:
- Text preprocessing (NLTK + spaCy)
- TF-IDF feature extraction
- Logistic Regression for classification

---

## 📁 Dataset

The project expects a CSV file named `dataset.csv` with the following columns:

- `review`: The movie review text
- `sentiment`: The sentiment label — either `positive` or `negative`

---

## 🔧 Preprocessing

Steps:
1. Lowercase conversion
2. Removal of HTML, punctuation, and numbers
3. Tokenization using NLTK
4. Stop word removal
5. Lemmatization using spaCy

---

## 🧠 Model

- **TF-IDF Vectorization** with max 5000 features
- **Logistic Regression** classifier (`liblinear` solver)
- Train/Test split: 80% / 20%

---

## 📈 Evaluation

- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix** (visualized with Seaborn)

---

## 🧪 How to Use

```bash
# Clone the repo
git clone https://github.com/your-username/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp

# Install required libraries
pip install -r requirements.txt

# Place your dataset.csv in the root folder

# Run the notebook
jupyter notebook Sentiment_Analysis.ipynb
