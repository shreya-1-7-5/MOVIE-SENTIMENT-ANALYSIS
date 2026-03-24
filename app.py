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
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    accuracies = joblib.load("accuracies.pkl")  
    y_test, preds = joblib.load("results.pkl") 
except:
    st.error(" Required files not found. Please upload all .pkl files.")
    st.stop()
st.set_page_config(page_title="Movie Sentiment Analysis", page_icon="🎬")
st.title(" Movie Sentiment Analysis with Visualization")
st.header(" Predict Sentiment")
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
st.header(" Model Accuracy Comparison")
fig1 = plt.figure()
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
st.pyplot(fig1)
best_model = max(accuracies, key=accuracies.get)
st.success(f"🏆 Best Model: {best_model}")
st.header("Confusion Matrix")
cm = confusion_matrix(y_test, preds)
fig2, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
st.pyplot(fig2)
