
'''
This is a collection of NLP analyses, including
    text cleaning
    most common words
    n-grams generation
    co-occurrence matrix generation. This is by using a plain Python method
    wordcloud generation
    topic modeling (using LDA)
    general text statistics
'''
'''
IMPORTING THE REQUIRED LIBRARIES
'''
import logging
logging.captureWarnings(True)
import numpy as np
import pandas as pd
from collections import defaultdict
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim_models
from math import log
import networkx as nx
from pyvis.network import Network
#import csv
from scipy.stats import entropy
'''
CREATING THE REQUIRED FUNCTIONS
'''
# cleans text
def txt_clean(word_list, stopwords_list, min_len):
    clean_words = []
    vocab = []
    for line in word_list:
        parts = line.strip().split()
        for word in parts:
            word_l = word.lower()
            if word_l not in stopwords_list:
                if word_l.isalpha():
                    if len(word_l) > min_len:
                        clean_words.append(word_l)
                        if word_l not in vocab:
                            vocab.append(word_l)
    return clean_words, vocab
# creates a list of the "num" most common elements/strings in an input list
def most_common(lst, num):
    data = Counter(lst)
    common = data.most_common(num)
    top_comm = []
    for i in range (0, num):
        top_comm.append (common [i][0])
    return top_comm
# creates a list of n-grams. Individual words are joined together by "_"
def ngram(text,grams):  
    n_grams_list = []
    count = 0
    for token in text[:len(text)-grams+1]:  
       n_grams_list.append(text[count]+' '+text[count+grams-1])
       count=count+1  
    return n_grams_list
def chunk_replacement(chunk_list, text):
    """
    Connects words chunks in a text by joining them with an underscore.
    
    :param chunk_list: word chunks
    :type chunk_list: list of strings/ngrams
    :param text: text
    :type text: string
    :return: text with underscored chunks
    :type: string
    """
    for chunk in chunk_list:
        text = text.replace(chunk, chunk.replace(' ', '_'))
    return text
# creates a matrix of co-occurrence
def co_occurrence(in_text, vocab, window_size, cooccurrence_min):
    d = defaultdict(int)
    for i in range(len(in_text)):
        token = in_text[i]
        next_token = in_text[i+1 : i+1+window_size]
        for t in next_token:
            key = tuple( sorted([t, token]) )
            d[key] += 1
    # formulate the dictionary into dataframe
    vocab = sorted(vocab) # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        if value <= cooccurrence_min:
            value = 0
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df
# creating and presenting the wordcloud
def cloud (words, stpwords, file_name):
    # Defining the wordcloud parameters
    wc = WordCloud(background_color = "white", max_words = 2000,
        stopwords = stpwords)
    # Generate words cloud
    wc.generate(words)
    # Store to file
    wc.to_file(file_name +'.png')
    # Show the cloud
    plt.imshow(wc)
    plt.axis('off')
    plt.show()
# extracting the topics using LDA
def topic_modeling(words, num_of_topics, file_name, algorithm='lda', 
file_out=True):
    """
    topic_modeling: Topic Modeling is a technique to extract the hidden topics from
large volumes of text.
    Latent Dirichlet Allocation(LDA) is a topic modelling technique. In LDA with 
gensim is employed to create a
    dictionary from the data and then to convert to bag-of-words corpus.
    pLDAvis is designed to help users interpret the topics in a topic model that 
has been fit to a corpus of text data.
    The package extracts information from a fitted LDA topic model to inform an 
interactive web-based visualization.
    :param words: List of words to be passed
    :param num_of_topics: Define the number of topics
    :param algorithm: 'lda' - the topic modelling technique
    :param file_out: If 'TRUE' generate an html for the LDA
    :return: list of topics
    """
    tokens = [x.split() for x in words]
    #print (words)
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]
    lda = LdaModel(corpus = corpus, num_topics = num_of_topics, id2word = 
dictionary)
    topic_lst = []
    print ('\nthe following are the top', num_topics, 'topics:')
    for i in range (0, len(lda.show_topics(num_of_topics))):
        topic_lst.append(lda.show_topics(num_of_topics)[i][1])
        print(lda.show_topics(num_of_topics)[i][1], '\n')
    if file_out == True:
        lda_display = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary, 
sort_topics=False)
        pyLDAvis.save_html(lda_display, 'LDA_' + file_name + '.html')
        #return (pyLDAvis.display(lda_display))
    return topic_lst
'''
MAIN PROGRAM
'''
# reading input files
input_file_name = 'NYT Data'
import pandas as pd
data = pd.read_csv('nyt_new.csv',low_memory=False)
abstract = data['abstract']
abstract_str = ""
for i in abstract:
  abstract_str += str(i)


stopwords_file = open('stopwords_en.txt','r', encoding='utf8')
# initializing lists
stopwords = []
txt_words = []
# populating the list of stopwords and the list of words from the text file
for word in stopwords_file:
    stopwords.append(word.strip())
# Updating the stopwords list
stopwords.extend(['new','people','said','year','maps','need','know','time','make','just' ])
for word in abstract_str.split():
    txt_words.append(word.strip())
# setting the minimum word length
min_word_len = 2
# setting the window of separation between words for the network creation
word_window = 1
# setting the minimum number of pair co-occurrence for the network creation
min_pair_coocc = 30
# setting the "n" for the n-grams generated
n_value = 2
# setting the number of elements to be similar
n_sim_elems = 10
# defining the number of topics to be extracted
num_topics = 5
print ('\n---This is an analysis for', input_file_name)
# cleaning the words and getting the list of unique words
clean_words, vocabulary = txt_clean(txt_words, stopwords, min_word_len)
all_words_string = ' '.join(clean_words)
# generating the n-grams and taking the top "n" elements (equal to "n_sim_elems")
ngrams_list = ngram(clean_words, n_value)
top_ngrams = most_common(ngrams_list, n_sim_elems)
text_chunked = chunk_replacement(top_ngrams, all_words_string)
text_chunked_lst = list(text_chunked.split(' '))
print ('\nthe following are the top', n_sim_elems, n_value, '-grams:\n', 
top_ngrams,'\n')
# generating the wordcloud
#   creating a string of words and top ngrams
all_stopwords_string = ' '.join(stopwords)
#   calling wordcloud generator
cloud_name = 'cloud_' + input_file_name
cloud (all_words_string, all_stopwords_string, cloud_name)
# extracting topics
topic_list = topic_modeling (text_chunked_lst, num_topics, input_file_name)
# general statistics on the input text
tot_num_words = len (text_chunked_lst)
unique_words_num = len (vocabulary)
# calculating the entropy in the text
words_counter = Counter(text_chunked_lst)
word_freq_lst = list(words_counter.values())
entropy_val = entropy(word_freq_lst, base = 10)
print ('\nthe following are some basic statistics on the input text')
print ('   total number of words:', tot_num_words)
print ('   total number of unique words:', unique_words_num)
print ('   total entropy in the text:', log(entropy_val, 10),'\n','(entropy is a measure of information rate)')
# creating the statistics table and wrting to a csv file
stat_dic = {'top_ngrams':top_ngrams, 'top_topics':topic_list, 
'num_words':str(tot_num_words), 'unique_words':str(unique_words_num), 
'entropy':str(entropy_val)}
stat_name = 'stats_' + input_file_name
with open(stat_name + '.csv', 'w') as f:
    for key in stat_dic.keys():
        f.write("%s, %s\n" % (key, stat_dic[key]))
print ('\n----this ends the process----\n')

