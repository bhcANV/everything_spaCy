import spacy
from spacy.tokens import Span

nlp = spacy.blank("en")

doc1 = nlp("iPhone X is coming")
doc1.ents = [Span(doc1, 0, 2, label="GADGET")]
doc2 = nlp("I need a new phone! Any tips?")
docs = [doc1, doc2]

random.shuffle(docs)
train_docs = docs[:len(docs) // 2]
dev_docs = docs[len(docs) // 2:]

train_docbin = DocBin(docs=train_docs)
train_docbin.to_disk("./train.spacy")

dev_docbin = DocBin(docs=dev_docs)
dev_docbin.to_disk("./dev.spacy")