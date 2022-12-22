# For this code, I used the vector spce code on canvas and made modification for our classification.
from gensim.models import Word2Vec
import pandas as pd
from datetime import datetime
import multiprocessing
from time import time
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot
from pyemd import emd
from sklearn.model_selection import train_test_split


def txt_clean(word_list, min_len, stopwords_list):
    """
    Performs a basic cleaning to a list of words.

    :param word_list: list of words
    :type: list
    :param min_len: minimum length of a word to be acceptable
    :type: integer
    :param stopwords_list: list of stopwords
    :type: list
    :return: clean_words list of clean words
    :type: lists

    """
    clean_words = []
    for line in word_list:
        parts = line.strip().split()
        for word in parts:
            word_l = word.lower().strip()
            if word_l.isalpha():
                if len(word_l) > min_len:
                    if word_l not in stopwords_list:
                        clean_words.append(word_l)
                    else:
                        continue

    return clean_words


'''
----------- Main
'''


# loading the stopwords and setting the minimum length for the words
stp_file = 'stopwords_en.txt'
stopwords_file = open(stp_file,'r', encoding='utf8')

# initializing the list of stopwords
stopwords_list = []

# populating the list of stopwords and the list of words from the text file
for word in stopwords_file:
    stopwords_list.append(word.strip())

word_min_len = 2

# printing starting time
start_time = datetime.now()
print ('\n-Starting the embedding process at {}'.format(start_time), '\n')

# reading a corpus from reviews dataset and loading it into a list
'''
Used this code to concatenate all the reviews into a text for the model.
with open("result.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())
            outfile.write('eof'.encode())
'''
txt_file = pd.read_csv('nyt_new.csv',low_memory=False)
txt_articles = txt_file['abstract'].to_numpy()
# creating a list of lists of cleaned words
data = pd.read_csv('nyt_new.csv',low_memory=False)
abstract = data['abstract']
abstract_str = ""
for i in abstract:
  abstract_str += str(i)

corpus = abstract_str.strip().replace('\n\n\n','\n').replace('\n\n','\n').replace('\n','#').replace('(','').replace(')','')
# creating a list from the corpus
#   it is a list of strings composed by words to be cleaned
#   each string is a "phrase"
corpus_lst = corpus.split("#")
    
corpus_lst_lst = []
for elem in corpus_lst:
    clean_elem = txt_clean(elem.split(), word_min_len, stopwords_list)
    corpus_lst_lst.append(clean_elem)

# setting the parameters for Word2Vec
cores = multiprocessing.cpu_count()
t = time()

# defining the parameters for the model
w2vec_model = Word2Vec(min_count=50, window=6,vector_size=20,sample=1e-5, alpha=0.5, negative=11, workers=cores-1)
# building the vocabulary for the model
w2vec_model.build_vocab(corpus_lst_lst)
# training the model
w2vec_model.train(corpus_lst_lst, total_examples=w2vec_model.corpus_count, epochs=200)

print('Time to develop and train the model: {} mins'.format(round((time() - t) / 60, 4)))

# extracting labels (that are the words in the vocabulary) and vectors from the model as numpy arrays
labels = np.asarray(w2vec_model.wv.index_to_key)
vectors = np.asarray(w2vec_model.wv.vectors)

print(w2vec_model.wv.most_similar('trump'))


data =data.dropna()
X = data['lead_paragraph']
y=data['type_of_material']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=20,train_size=0.7,shuffle=True)

words = set(w2vec_model.wv.index_to_key )
X_train_vect = np.array([np.array([w2vec_model.wv[i] for i in str(ls) if i in words])
                         for ls in X_train])
X_test_vect = np.array([np.array([w2vec_model.wv[i] for i in str(ls) if i in words])
                         for ls in X_test])

# Compute sentence vectors by averaging the word vectors for the words contained in the sentence
X_train_vect_avg = []
for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        X_train_vect_avg.append(np.zeros(100, dtype=float))
        
X_test_vect_avg = []
for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(100, dtype=float))

# Instantiate and fit a basic Random Forest model on top of the vectors
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_model = rf.fit(X_train_vect_avg, y_train.values.ravel())

# Use the trained model to make predictions on the test data
y_pred = rf_model.predict(X_test_vect_avg)
from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, y_pred,average='micro')
recall = recall_score(y_test, y_pred,pos_label='Pos',average='micro')
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))