# import import_data as data
import os
import pandas as pd
# import gensim
import numpy as np
import spacy
import clusterize

nlp = spacy.load('pt_core_news_sm')
# python -m spacy download pt_core_news_sm

reviews = pd.read_excel('data1.xlsm',sheet_name='Result', index_col=0)
# columns Index(['Review ID', 'Location Name', 'Group Name', 'Rating', 'Content', 'Data','Source'])
dados = reviews.dropna()


def clean_doc(doc):
    text = [token.lemma_ for token in doc if
            token.text != "" and token.text != " " and token.is_punct == False and token.is_stop == False]
    return text

def clean_to_vectors(doc):
    text = [token.vector for token in doc if
            token.text != "" and token.text != " " and token.is_punct == False and token.is_stop == False]
    return text

def comment_to_vector(doc):
    return doc.vector

def comment_to_vector_norm(doc):
    return doc.vector_norm

def send_dados_to_picle(dados):
    dados.to_pickle('dados.pkl')

def get_word_vector (dados):
    # uso de spacy
    dados['Docs'] = dados['Content'].apply(lambda x: nlp(x)) # comentarios tokenizados pelo spacy
    dados['Docs_clean'] = dados['Docs'].apply(lambda x: clean_doc(x)) # cada linha sao palavras lematizadas, sem pontuacao e stopwords
    dados['Docs_vector'] = dados['Docs'].apply(lambda x: clean_to_vectors(x)) # cada linha sao vetores das palavras lematizadas, sem pontuacao e stopwords, de cada comentario
    # send_dados_to_picle(dados)
    to_cluster_vector = []
    # soma todos os vetores palavras do commentario
    for comment in dados['Docs_vector']:
        for word_vector in comment:
            to_cluster_vector.append(word_vector)

    t2 = np.stack(to_cluster_vector, axis=0)
    return t2

def get_comment_vector (dados):
# uso de spacy
    dados['Docs'] = dados['Content'].apply(lambda x: nlp(x))  # comentarios tokenizados pelo spacy
    dados['Docs_clean'] = dados['Docs'].apply(
        lambda x: clean_doc(x))  # cada linha sao palavras lematizadas, sem pontuacao e stopwords
    dados['Comment_vector'] = dados['Docs'].apply(lambda x: comment_to_vector(
        x))  # cada linha sao vetores das palavras lematizadas, sem pontuacao e stopwords, de cada comentario
    # send_dados_to_picle(dados)
    to_cluster_vector = []
    # soma todos os vetores palavras do commentario
    for comment in dados['Comment_vector']:
        to_cluster_vector.append(comment)
    t2 = np.stack(to_cluster_vector, axis=0)
    return t2

def get_comment_vector_norm (dados):
# uso de spacy
    dados['Docs'] = dados['Content'].apply(lambda x: nlp(x))  # comentarios tokenizados pelo spacy
    dados['Docs_clean'] = dados['Docs'].apply(
        lambda x: clean_doc(x))  # cada linha sao palavras lematizadas, sem pontuacao e stopwords
    dados['Comment_vector_norm'] = dados['Docs'].apply(lambda x: comment_to_vector_norm(
        x))  # cada linha sao vetores das palavras lematizadas, sem pontuacao e stopwords, de cada comentario
    # send_dados_to_picle(dados)
    to_cluster_vector = []
    # soma todos os vetores palavras do commentario
    for comment in dados['Comment_vector_norm']:
        to_cluster_vector.append(comment)


    t2 = np.stack(to_cluster_vector, axis=0)
    return t2


#clusterize.clusterize_share(get_comment_vector(dados))
#clusterize.clusterize_structure(get_comment_vector(dados))


