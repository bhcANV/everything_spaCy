import spacy
import sys
import json

def load_data(file):
    with open(file, "r") as f:
        text = f.read()
    return(text)

def save_data(file, data):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def get_entities(doc):
    for sentence in doc.sents:



file_name = sys.argv[1]
nlp = spacy.load("en_core_web_sm")
doc = nlp(load_data(file_name))




