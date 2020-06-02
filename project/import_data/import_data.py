
import pandas as pd
import nltk
import pandas as pd
import string
import re

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)

stopwords = nltk.corpus.stopwords.words('portuguese')
# add new stop words
new_stopwords = ['s']
stopwords.extend(new_stopwords)
stemmer = nltk.stem.RSLPStemmer()

reviews = pd.read_excel('data1.xlsm',sheet_name='Result', index_col=0)
# columns Index(['Review ID', 'Location Name', 'Group Name', 'Rating', 'Content', 'Data','Source'])

reviews_comment = reviews.dropna()
# print(reviews_comment)
# print(reviews_comment['Group Name'].unique())

# extract external data
def extract_data():
    reviews = pd.read_excel('data1.xlsm',sheet_name='Result', index_col=0)
    return reviews.dropna()

# remove punctuation from text(sentece)
def remove_punctuation(text):
    text_no_punct = "".join([char for char in text if char not in string.punctuation])
    return text_no_punct


# tokenize text(sentece)
def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens


# remove stopwords in tokenized text(sentece)
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopwords]
    text = [word for word in text if word != ""]    # remove empty element from list
    return text


# stemming tokenized text(sentence)
def stemming(tokenized_text):
    text = [stemmer.stem(word) for word in tokenized_text]
    return text


# remove punctuations, stopwords and stemm text, return text
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    tokens = [word for word in tokens if word != ""]
    text = [stemmer.stem(word) for word in tokens if word not in stopwords]
    return text


#  count punctuations, return pct of punctuations in text
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100


#  count bebdida, return pct of punctuations in text
def count_cerveja(text):
    count = sum([1 for word in text if word in ['bebdida','cerveja']])
    return count

def count_ambiente(text):
    count = sum([1 for word in text if word=='ambiente'])
    return count


def count_atendimento(text):
    count = sum([1 for word in text if word=='atendimento'])
    return count