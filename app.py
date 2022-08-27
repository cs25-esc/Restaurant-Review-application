import streamlit as st
import pickle

import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

stw = PorterStemmer()


count_vectorizer = pickle.load(open('v.pkl','rb'))
model = pickle.load(open('classifier.pkl','rb'))


def clean(review):
    review = review.lower()
    review = re.sub(r'[^\w\s]', "", review)
    review = nltk.word_tokenize(review)
    words = [i for i in review if i not in stop_words]
    wer = [stw.stem(i) for i in words]

    return " ".join(wer)

st.title("Restaurant - Review Classifier")

review = st.text_input("Enter your review")

if st.button('Classify'):

    #preprocess
    cleaned_text = clean(review)

    #vectorize
    cv = count_vectorizer.transform([cleaned_text])

    #model feeding
    result = model.predict(cv)[0]

    #predict the review sentiment
    if result == 1:
        st.header("Thank you please visit again")

    else:
        st.header("Sorry, for the inconveinence we will try to imrove")
