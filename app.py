import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Title UI
st.markdown(
    """
    <h1 style='text-align: center;'>📰 Fake News Detection System</h1>
    <p style='text-align: center;'>AI-powered NLP model using TF-IDF & Logistic Regression</p>
    """,
    unsafe_allow_html=True
)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

st.markdown("---")

input_text = st.text_area("Enter News Article Text")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(input_text)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)
        probabilities = model.predict_proba(vector)

        real_conf = probabilities[0][1] * 100
        fake_conf = probabilities[0][0] * 100

        if prediction[0] == 1:
            st.success("✅ REAL News")
            st.write(f"Confidence: {real_conf:.2f}%")
        else:
            st.error("❌ FAKE News")
            st.write(f"Confidence: {fake_conf:.2f}%")

st.markdown("---")
st.info("Model trained on political news dataset using Machine Learning.")