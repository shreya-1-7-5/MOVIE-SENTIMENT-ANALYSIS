import streamlit as st
import joblib
import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Load model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Dummy accuracy values (replace with your real ones if saved)
accuracies = {
    "Naive Bayes": 0.85,
    "Logistic Regression": 0.89,
    "SVM": 0.91
}

st.title("🎬 Movie Sentiment Analysis with Visualization")
st.header("🔍 Predict Sentiment")

review = st.text_area("Enter your review:")

if st.button("Analyze"):
    if review.strip():
        clean = clean_text(review)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😠 Negative Review")
    else:
        st.warning("Please enter a review")

st.header("Model Accuracy Comparison")

fig1 = plt.figure()
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")

st.pyplot(fig1)

st.header("Confusion Matrix")

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 0, 1]

cm = confusion_matrix(y_true, y_pred)

fig2, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

st.pyplot(fig2)
