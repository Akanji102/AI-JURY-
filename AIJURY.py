
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('stopwords')
nltk.download('punkt')


model = joblib.load('scotus_prediction.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


def preprocess_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)


st.set_page_config(
    page_title="⚖️ AI Jury",
    page_icon="⚖️",
    layout="centered",
)


st.title("⚖️ AI Jury: Judicial Outcome Predictor")
st.markdown("""
Welcome to **AI Jury**, an intelligent system that predicts the likely **court outcome** based on the presented case facts.  
This model analyzes historical judicial patterns and returns a likely **winner** and **court disposition**.
""")


case_facts = st.text_area(
    "📝 Present the facts of the case below:",
    height=200,
    placeholder="e.g., The defendant was accused of violating environmental regulations after repeated non-compliance with EPA standards..."
)


if st.button("🔍 Predict Outcome"):
    if not case_facts.strip():
        st.warning("Please enter the facts of the case before prediction.")
    else:

        processed_input = preprocess_text(case_facts)
        processed_input_vectorized = vectorizer.transform([processed_input])

        prediction = model.predict(processed_input_vectorized)[0]


        if prediction == 1:
            winner_text = "✅ **First Party is likely to WIN the case.**"
            disposition_text = "🏛️ **Disposition:** Judgment Affirmed or Upheld."
        elif prediction == 2:
            winner_text = "⚖️ **Case is likely to be DISMISSED.**"
            disposition_text = "📜 **Disposition:** Case Dismissed or Dropped."
        else:
            winner_text = "❌ **First Party is likely to LOSE the case.**"
            disposition_text = "⚖️ **Disposition:** Judgment Reversed or Overturned."


        st.success(winner_text)
        st.info(disposition_text)

st.markdown("---")
st.caption("Developed by **Fawole Joshua Ajibola** — AI for Justice, Transparency, and Fairness in the legal realm")
