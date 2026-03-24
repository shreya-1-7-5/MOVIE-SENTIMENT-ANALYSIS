import gradio as gr
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
    raise Exception("Required .pkl files not found!")
def predict_sentiment(review):
    if review.strip() == "":
        return " Please enter a review", None, None

    clean = clean_text(review)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]

    result = "😊 Positive Review" if pred == 1 else "😠 Negative Review"

    fig1 = plt.figure()
    plt.bar(accuracies.keys(), accuracies.values())
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Comparison")
    cm = confusion_matrix(y_test, preds)

    fig2, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)

    return result, fig1, fig2
with gr.Blocks() as app:
    gr.Markdown("# 🎬 Movie Sentiment Analysis with Visualization")

    review_input = gr.Textbox(label="Enter your review")

    output_text = gr.Textbox(label="Prediction")
    output_graph1 = gr.Plot(label="Accuracy Comparison")
    output_graph2 = gr.Plot(label="Confusion Matrix")

    analyze_btn = gr.Button("Analyze")

    analyze_btn.click(
        fn=predict_sentiment,
        inputs=review_input,
        outputs=[output_text, output_graph1, output_graph2]
    )
app.launch()
