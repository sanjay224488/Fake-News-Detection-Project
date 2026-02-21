# 📰 Fake News Detection System

An AI-powered Fake News Detection system built using Machine Learning and Natural Language Processing (NLP).  
This project classifies news articles as **Real** or **Fake** using TF-IDF and Logistic Regression.

---

##  Live Demo

👉 [Click Here to Try the App](https://fake-news-detection-project-yvzmtuw7mda9rck6okr7dv.streamlit.app)

---

##  Project Overview

Fake news spreads rapidly across digital platforms, influencing public opinion and creating misinformation.  
This project aims to automatically classify news articles as Real or Fake using text-based machine learning techniques.

---

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- GitHub

---

##  Dataset

The dataset consists of:

- **True.csv** – Real news articles
- **Fake.csv** – Fake news articles

Each record contains:
- Title
- News Content
- Subject
- Date

Real news labeled as **1**  
Fake news labeled as **0**

---

##  Methodology

1. Data Loading and Labeling
2. Text Preprocessing (Lowercasing, Stopword Removal, Cleaning)
3. Feature Extraction using TF-IDF (Unigrams + Bigrams)
4. Train-Test Split (80/20)
5. Model Training using Logistic Regression
6. Model Evaluation (Accuracy, Precision, Recall, F1-score)
7. Deployment using Streamlit

---

##  Model Performance

- Accuracy: **~99%**
- Balanced Logistic Regression
- Improved TF-IDF with 10,000 features

---

##  How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/sanjay224488/Fake-News-Detection-Project.git
