import spacy

text = "teste pacote spacy."
nlp = spacy.load('pt_core_news_sm')

doc = nlp(text)
for token in doc:
	print(token.vector)
	print(token.vector_norm)
