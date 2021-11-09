import pandas as pd 
import numpy as np 
import pickle
from nltk.stem.porter import *
import re
import gensim
from gensim.models import Doc2Vec
from tqdm import tqdm
from sklearn import utils

#splitter = re.compile('[^a-zA-Z0-9]')

def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False

def separate_words(text, min_word_return_size):
    """
    Utility function to return a list of all words that are have a length greater than a specified number of characters.
    @param text The text that must be split in to words.
    @param min_word_return_size The minimum no of characters a word must have to be included.
    """
    splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    words = []
    for single_word in splitter.split(text):
        current_word = single_word.strip().lower()
        #leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
        if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
            words.append(current_word)
    return words

def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    sentence_delimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
    sentences = sentence_delimiters.split(text)
    return sentences
def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    return stop_words

def build_stop_word_regex(stop_word_file_path):
    stop_word_list = load_stop_words(stop_word_file_path)
    stop_word_regex_list = []
    for word in stop_word_list:
        word_regex = r'\b' + word + r'(?![\w-])'  # added look ahead for hyphen
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern
stoppath = "SmartStoplist.txt"  #SMART stoplist misses some of the lower-scoring keywords in Figure 1.5, which means that the top 1/3 cuts off one of the 4.0 score words in Table 1.1
stopwordpattern = build_stop_word_regex(stoppath)


def generate_candidate_keywords(sentence_list, stopword_pattern):
    phrase_list = []
    for s in sentence_list:
        tmp = re.sub(stopword_pattern, '|', s.strip())
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if phrase != "":
                phrase_list.append(phrase)
    return phrase_list
def generate_seperate_words(phraseList):
    wordList=[]
    for phrase in phraseList:
        word_list = separate_words(phrase, 0)
        wordList+=word_list
    return wordList
def sent2word(sentence):
    sentenceList=split_sentences(sentence)
    phraseList = generate_candidate_keywords(sentenceList, stopwordpattern)
    wordlist=generate_seperate_words(phraseList)
    return wordlist
    




review_df=pd.read_pickle('gnn_review_df.pkl')
print(review_df.columns.values.tolist())
'''
['review_id', 'user_id', 'business_id', 'review_stars', 
'text', 'useful', 'funny', 'cool', 'label', 'mask']
'''


review_text=review_df['text'].tolist()
documents=[]
for idx,review in enumerate(tqdm(review_text)):
    #print(idx)
    words=sent2word(review)
    # print(words)
    # print(len(words))
    # exit(0)
    
    documents.append(gensim.models.doc2vec.TaggedDocument(words, [str(idx)]))
model = Doc2Vec(vector_size=64, window=15, min_count=1, sample=1e-5, negative=5, dm=0, workers=1) 
model.build_vocab([x for x in tqdm(documents)])
for epoch in range(30):
    model.train(utils.shuffle([x for x in tqdm(documents)]),total_examples=len(documents),epochs=1)

model.save('review_d2v.model')
