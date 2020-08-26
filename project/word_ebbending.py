
# Project library
# import project.import_data as import_data
import import_data

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
		to_cluster_vector.append(comment/np.linalg.norm(comment))
	comment_matrix = np.stack(to_cluster_vector, axis=0)

	return comment_matrix,dados

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

# def get_comment_vector_angle_v2(dados):
#     comment_matrix_angle = []
#     comment_matrix, base = get_comment_vector(dados)
#     for i_comment_matrix in range(len(comment_matrix)):
#         comment_matrix_angle.append(list())
#         # -1: last dimension
#         # print('comment: ', i_comment_matrix, '\n')
#         # for i_dimension in range(len(comment_matrix[i_comment_matrix])-1):
#         for i_dimension in range(len(comment_matrix[i_comment_matrix])-1):
#             # print('dimension: ', i_dimension, '\n')
#             # +1: i_next_dimension inicialize in zero
#             for i_next_dimension in range(len(comment_matrix[i_comment_matrix]) - (i_dimension +1)):
#             # for i_next_dimension in range(len(comment_matrix[i_comment_matrix])):    
#                 unit_op_cat = comment_matrix[i_comment_matrix][i_dimension]/np.linalg.norm(comment_matrix[i_comment_matrix][i_dimension])
#                 unit_adj_cat = comment_matrix[i_comment_matrix][i_dimension + (i_next_dimension+1)]/np.linalg.norm(comment_matrix[i_comment_matrix][i_dimension + (i_next_dimension + 1)])
#                 print(unit_op_cat)
#                 print(unit_op_cat)
#                 dot_product = np.dot(unit_op_cat,unit_adj_cat)
#                 # angle = np.arccos(dot_product)
#                 comment_matrix_angle[i_comment_matrix].append(dot_product)
#                 # tangent = comment_matrix[i_comment_matrix][i_dimension]/comment_matrix[i_comment_matrix][i_dimension - (i_next_dimension + 1)]
#                 # comment_matrix_angle[i_comment_matrix].append(tangent)
    
#     return comment_matrix_angle, base

# def get_comment_vector_angle(dados):
#     comment_matrix_angle = []
#     ref = []
#     comment_matrix, base = get_comment_vector(dados)
#     for i in range(96):
#         ref.append(1)
#     for i_comment_matrix in range(len(comment_matrix)):
#         # comment_matrix_angle.append(list())
#         # -1: last dimension
#         # print('comment: ', i_comment_matrix, '\n')
#         # for i_dimension in range(len(comment_matrix[i_comment_matrix])-1):
#         # for i_dimension in range(len(comment_matrix[i_comment_matrix])):  
#         norm = comment_matrix[i_comment_matrix]/np.linalg.norm(comment_matrix[i_comment_matrix])
#         print(norm)
#         dot_product = np.dot(norm,ref)
#                 # angle = np.arccos(dot_product)
#         comment_matrix_angle.append(dot_product)
#                 # tangent = comment_matrix[i_comment_matrix][i_dimension]/comment_matrix[i_comment_matrix][i_dimension - (i_next_dimension + 1)]
#                 # comment_matrix_angle[i_comment_matrix].append(tangent)
    
#     return comment_matrix_angle, base
