import re
import pandas as pd
import bs4
import requests
import spacy
import DataReader
from scipy.sparse import dok_matrix
from spacy import displacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.naive_bayes import MultinomialNB
nlp = spacy.load('en_core_web_sm')
from sklearn import metrics
import numpy as np
from spacy.matcher import Matcher
from spacy.tokens import Span

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm


class MostCommon:
    def __init__(self):
        self.x_test, self.x_train, self.y_test, self.y_train = DataReader.split_sheet_into_test_training_per_word()
        self.id = {}
        count = 0
        for w in self.x_train.iloc[:,0]:
            if w not in self.id:
                self.id[w] = count
                count += 1

    def prnt(self):
        # self.x_train.iloc[:,1] = TreebankWordDetokenizer().detokenize(self.x_train.iloc[:,1])
        for line in self.x_train.iloc[:,1]:
            line = TreebankWordDetokenizer().detokenize(line)
            for tok in nlp(line):
                print(tok.text, "...", tok.dep_)

    def createFeatures(self, data):
        features = dok_matrix((len(data), len(self.id)))
        for i in range(len(data)):
            line = data.iloc[i, 1]
            line = TreebankWordDetokenizer().detokenize(line)
            for tok in nlp(line):
                if tok.text in self.id:
                    features[i, self.id[tok.text]] = tok.dep
        return features


ms = MostCommon()
f = ms.createFeatures(ms.x_train)
l = np.ravel(ms.y_train)
testf = ms.createFeatures(ms.x_test)
testl = np.ravel(ms.y_test)

# train the model on the features
clf = MultinomialNB()
clf.fit(f, l)

# predict the labels for the test data
predicted = clf.predict(testf)
# output your results
print(metrics.classification_report(testl, predicted))
