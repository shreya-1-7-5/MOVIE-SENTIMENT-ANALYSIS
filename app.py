import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
st.set_page_config(page_title="Movie Sentiment Analysis", page_icon="🎬")
st.title("🎬 Movie Sentiment Analysis")
st.write("Enter a movie review and get sentiment prediction.")
review = st.text_area("Enter your review here:")
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        clean = clean_text(review)
        vec = vectorizer.transform([clean])
        prediction = model.predict(vec)[0]
        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😠 Negative Review")
