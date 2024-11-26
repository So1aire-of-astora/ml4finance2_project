import numpy as np
import pandas as pd
import re
import nltk
import gensim
from gensim.models import Word2Vec
# nltk.download("stopwords")

class Features:
    def __init__(self, body_path, stance_path):
        '''
        TODO:
        0. Merge: SELECT * FROM train_stances LEFT JOIN train_bodies ON train_stances.body_id = train_bodies.body_id 
        1. Preperation
            1.1 Normalize the texts (to lowercase, remove extra spaces and punctuation, tokenize)
            1.3 prepare the corpus based on the whole training set, i.e. pairs of headline and body. 
        2. Generate word embeddings, using word2vec in our case.
            2.1 not necessary to remove stop words (the embeddings for stop words can capture contextual relationships that might still be relevant)
            2.2 Pretrained model (GOogle news, gensim) or train our model using the dataset?
            2.3 The embedding vector of a n-gram is the avg of evary word. i.e. {"covid-19 vaccine": [.1, .2, .4], "vaccine approved": [-.5, 1.4, -.6]}
        3. TF-IDF feature engineering
            3.1 use the corpus to fit a tf-idf vectorizor. it should assign a tf-idf score to each n-gram, i.e. {"covid-19 vaccine": .4, "vaccine approved": .6}
                ** Question: what if new n-grams appear in the test set? by default, the score is set to zero. but important info could be ignored by doing so. **
                ** can remove stop words when initializing the vectorizer **
            3.2 the TF-IDF vector of each sample has a length of #(unique n-grams) in the corpus. usually sparse, with high dimension, may cause vanishing gradient. Solutions needed.
        4. Joint only features we can consider:
            4.1 Jaccard similarity (n-grams overlap): suppose A, B are the set of n-grams in the body & the headline. J(A, B) = |A and B| / |A or B| (# of common n-grams over # of unique n-grams accross both texts)
            4.2 cosine similarity of word2vec embeddings: Similarity = (headline_vec * body_vec) / [norm(headline_vec) * norm(body_vec)]
            4.3 cosine similarity of tf-idf vectors
            4.4 cosine similarity of tf-idf **score** weighted embedding vectors:
                4.4.1 weight word2vec embeddings for each n-gram in step 2 by its tf-idf score. just multiply them.
                4.4.2 the embedding of the headline/body is the average of the weighted embeddings for all n-grams in the headline/body **those can be individual features**
                4.4.3 compute similarity score using cosine similarity
        5. Other Individual/joint features we can consider:
            - average word2vec embedding
            - count features: sklearn
            - word length features: min, max, avg
            - lexical diversity: unique words / total words
            - sentiment: pretrained models, VADER, BERT (transformer-based), or train our own sentiment model
                ** Keep all the stop words and punctuations in the aforementioned pretrained models, because the embeddings for those words still carry useful semantic or syntactic information **
        '''
    def merge(self, body_path, stance_path):
        '''
        TODO 0
        '''
    
    @staticmethod
    def normalize_text(text, rm_stopword):
        '''
        TODO 1.1
        '''
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) # remove punctuation
        tokens = nltk.tokenize.word_tokenize(text)
        if rm_stopword:
            stop_words = set(nltk.corpus.stopwords.words('english'))
        else:
            stop_words = set()
        tokens = [word for word in tokens if word not in stop_words]
        return tokens
    



def main():
    training_set = Features(body_path = "./fnc-1-master/train_bodies.csv", stance_path = "./fnc-1-master/train_stances.csv")


if __name__ == "__main__":
    main()