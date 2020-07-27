from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from project.import_data import import_data as get_data
from project.word_ebbending import *
import nltk
import re
import string

reviews = pd.read_excel('data1.xlsm',sheet_name='Result',index_col=0)
dados = reviews.dropna().reset_index()
print(dados)

stopwords = nltk.corpus.stopwords.words('portuguese')
# add new stop words
new_stopwords = ['s']
stopwords.extend(new_stopwords)
stemmer = nltk.stem.RSLPStemmer()

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    tokens = [word for word in tokens if word != ""]
    text = " ".join([stemmer.stem(word) for word in tokens if word not in stopwords])

    return text




def get_matrix(docs,n=1,m=2):
    cv = CountVectorizer(docs, ngram_range=(n, m), preprocessor=clean_text)  # ,min_df=0.05)
    count_vector = cv.fit_transform(docs)
    print(cv.vocabulary_)  # devolve a palavra e sua posicao
    print(cv.get_feature_names())  # manda as palavras
    count_vect_df = pd.DataFrame(count_vector.todense(), columns=cv.get_feature_names())
    # return cv.vocabulary_,cv.get_feature_names() # retorna as palavras e identificadas
    return count_vect_df




def get_tdif_score(docs,n=1,m=2):
    cv = CountVectorizer(docs, ngram_range=(n, m), preprocessor=clean_text)  # ,min_df=0.05)
    count_vector = cv.fit_transform(docs)

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(count_vector)

    tf_idf_vector=tfidf_transformer.transform(count_vector)
    feature_names = cv.get_feature_names()

    # #get tfidf vector for first document
    # first_document_vector=tf_idf_vector[0]
    # df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
    aux = df = pd.DataFrame(tf_idf_vector[0].T.todense(), index=feature_names, columns=["tfidf"+str(0)])
    aux.sort_values(by=["tfidf"+str(0)], ascending=False, inplace=True)
    i=1
    max = tf_idf_vector.shape[0]
    print(f'max = {max}')
    while(i<max):

        df = pd.DataFrame(tf_idf_vector[i].T.todense(), index=feature_names, columns=["tfidf"+str(i)])
        df.sort_values(by=["tfidf"+str(i)], ascending=False, inplace=True)
        aux = pd.concat([aux,df],axis=1)
        i+=1

    return aux
#print the scores

docs = dados['Content'].tolist()
print(docs)
print(get_tdif_score(docs))