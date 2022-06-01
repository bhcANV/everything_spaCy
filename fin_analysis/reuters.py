import spacy
import pandas as pd
from spacy import displacy

df = pd.read_csv("stocks.tsv", sep="\t")
symbols = df.Symbol.tolist()
companies = df.CompanyName.tolist()
print(symbols[:10],'\n')

df2 = pd.read_csv("indexes.tsv", sep="\t")
indexes = df2.IndexName.tolist()
index_symbols = df2.IndexSymbol.tolist()

df3 = pd.read_csv("stock_exchanges.tsv", sep="\t")
exchanges = df3.ISOMIC.tolist()+df3["Google Prefix"].tolist()+df3.Description.tolist()
print(exchanges[:10],'\n')

nlp = spacy.blank("en")
ruler = nlp.add_pipe("entity_ruler")
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
stops = ["two"]
patterns = []

for symbol in symbols:
    patterns.append({"label": "STOCK", "pattern": symbol})
    for l in letters:
        patterns.append({"label": "STOCK", "pattern": symbol+f".{l}"})

for company in companies:
    if company not in stops:
        patterns.append({"label": "COMPANY", "pattern": company})

for index in indexes:
    patterns.append({"label": "INDEX", "pattern": index})
    words = index.split()
    patterns.append({"label": "INDEX", "pattern": " ".join(words[:2])})

for index in index_symbols:
    patterns.append({"label": "INDEX", "pattern": index})

for e in exchanges:
    patterns.append({"label": "STOCK_EXCHANGE", "pattern": e})
    

ruler.add_patterns(patterns)



with open("reuters.txt", "r") as f:
    text = f.read()
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)

