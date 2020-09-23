
# Project library
import project.import_data as import_data
# import import_data

# External library
import pandas as pd
# import gensim
import numpy as np
import spacy

nlp = spacy.load('pt_core_news_sm')
# python -m spacy download pt_core_news_sm

# nlp.Defaults.stop_words |=

def clean_doc(doc):
    text = [token.lemma_ for token in doc if
            token.text != "" and token.text != " " and token.is_punct == False and token.is_stop == False]
    # print(text)
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

    word_matrix = np.stack(to_cluster_vector, axis=0)
    return word_matrix

def get_comment_vector (dados):
    # caso 1 - vetor 96

	# print(dados['Location Name'])
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
        # to_cluster_vector.append(comment/np.linalg.norm(comment))
	comment_matrix = np.stack(to_cluster_vector, axis=0)

	return comment_matrix,dados


def get_comment_vector_div_norm(dados):
    # caso 2 - alteracao do dividido pela normal

    # print(dados['Location Name'])
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
        to_cluster_vector.append(comment / np.linalg.norm(comment))
    # to_cluster_vector.append(comment/np.linalg.norm(comment))
    comment_matrix = np.stack(to_cluster_vector, axis=0)

    return comment_matrix, dados

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

    comment_matrix = np.stack(to_cluster_vector, axis=0)
    
    return comment_matrix

def get_comment_vector_cos(dados):
    # nao vai usar
    comment_matrix_cos = []
    ref = []
    comment_matrix, base = get_comment_vector(dados)
    for i in range(96):
        ref.append(1)
    for i_comment_matrix in range(len(comment_matrix)):
        cos = np.vdot(comment_matrix[i_comment_matrix],ref)/((np.linalg.norm(comment_matrix[i_comment_matrix]))*np.linalg.norm(ref))
        comment_matrix_cos.append(cos)
    return comment_matrix_cos, base

def get_comment_vector_cos_v2(dados):
    # caso 3 par a par
    ref = [1,1]
    # Inicialização da matrix de cossenos dos comentários
    comment_matrix_cos = []
    # Variávies de matrix de comentários já vetorizados (comment_matrizx) e data frame da base (base)   
    comment_matrix, base = get_comment_vector(dados)
    # Para cada comentário vetorizado faça:
    for i_comment_matrix in range(len(comment_matrix)):
        # Preenchimento de 1 vetor vazio por comentário
        comment_matrix_cos.append(list())
        # Para cada dimensão de comentário faça:
        # -1: vetor começa em zero.
        for i_dimension in range(len(comment_matrix[i_comment_matrix])-1):
            pair_vector = [ comment_matrix[i_comment_matrix][i_dimension], comment_matrix[i_comment_matrix][i_dimension + 1]]
            cos = np.vdot(pair_vector,ref)/((np.linalg.norm(pair_vector))*np.linalg.norm(ref))
            comment_matrix_cos[i_comment_matrix].append(cos)

    return comment_matrix, dados

def get_comment_vector_cos_v3(dados):
    # caso 4 comb de vetor
    ref = [1,1]
    # Inicialização da matrix de cossenos dos comentários
    comment_matrix_cos = []
    # Variávies de matrix de comentários já vetorizados (comment_matrizx) e data frame da base (base)   
    comment_matrix, base = get_comment_vector(dados)
    # Para cada comentário vetorizado faça:
    for i_comment_matrix in range(len(comment_matrix)):
        # Preenchimento de 1 vetor vazio por comentário
        comment_matrix_cos.append(list())
        # Para cada dimensão de comentário faça:
        # -1: vetor começa em zero + (-1) menos o último valor
        for i_dimension in range(len(comment_matrix[i_comment_matrix])-2):
            # Para cada dimensão de index maior que i_dimension é construído um vetor (valor1, valor2) 
            second_dimension = i_dimension + 1
            while(second_dimension <= len(comment_matrix[i_comment_matrix])-1):
                pair_vector = [ comment_matrix[i_comment_matrix][i_dimension], comment_matrix[i_comment_matrix][second_dimension]]
                cos = np.vdot(pair_vector,ref)/((np.linalg.norm(pair_vector))*np.linalg.norm(ref))
                comment_matrix_cos[i_comment_matrix].append(cos)
                second_dimension = second_dimension + 1
    return comment_matrix, dados

# comment_matrix_cos = get_comment_vector_cos_v3(import_data.import_data())
# print(comment_matrix_cos,'\n')
# print(len(comment_matrix_cos),'\n')
