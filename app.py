import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

# Download only stopwords
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

def transform_text(text):
    text = text.lower()

    # Tokenize text
    text = wordpunct_tokenize(text)

    y = []

    # Keep alphanumeric words
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter your message")

if st.button("Predict"):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")
