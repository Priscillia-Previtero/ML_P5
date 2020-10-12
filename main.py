from fastapi import FastAPI
from pydantic import BaseModel

import re
from bs4 import BeautifulSoup
import nltk

import pickle
import custom_estimator

app = FastAPI()

#Chargement des stopwords
try:
    stopwords = nltk.corpus.stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')
wnLemma = nltk.stem.WordNetLemmatizer()
lemma_stopwords = [wnLemma.lemmatize(y) for y in stopwords]
stopwords = set(stopwords + lemma_stopwords + ['.', '', '-', '...', '--', '..', '+'])

#Chargement du modèle
loaded_model = pickle.load(open('final_model.sav', 'rb'))


def clean_input_from_html(input_text):
    basic_regex = '[ \n\r\t]'
    text= re.sub(basic_regex, " ", input_text)     
    text = BeautifulSoup(text, 'html.parser').get_text() 
    return text

def tokenize(text):
    text = text.replace("n't", " not") 
    tokens = [t.lower() for t in nltk.word_tokenize(text)] 
    pattern = '[^a-z.+-]+' # on garde + . et - pour les motys type 'my-sql', '.net', 'c++'
    tokens = [re.sub(pattern, '', t) for t in tokens if re.sub(pattern, '', t) != '']  
    tokens = [wnLemma.lemmatize(t) for t in tokens if t not in stopwords]
    return tokens


@app.put("/{text}")
def update_item(text: str):
    #Nettoyage
    text = clean_input_from_html(text)
    #Tokenisation + normalisation
    tokens = tokenize(text)

    X_test = ' '.join([y for y in tokens]) 
    output = loaded_model.predict([X_test])
    return output[0]

