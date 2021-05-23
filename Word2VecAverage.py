import numpy as np
import pandas as pd
import gensim.models
import gensim.downloader


class Word2VecAverage:
    def __init__(self):
        self.model = None
        self.word_vectors = None

    def train_word2vec(self, data=None):
        if data is None:
            self.model = gensim.downloader.load("glove-wiki-gigaword-50")
            # self.word_vectors = self.model.wv
            return

        self.model = gensim.models.Word2Vec(sentences=data)
        # self.word_vectors = self.model.wv
        return

    def create_feature_set(self, data):
        vector_length = self.model.vector_size
        features = np.zeros((len(data), vector_length))
        for i in range(len(data)):
            sentence = data.iloc[i]["line"]
            sentence_average = np.zeros(vector_length)
            count = 0

            for w in sentence:
                if w in self.model:
                    sentence_average += self.model[w]
                    count += 1

            if count > 0:
                sentence_average = sentence_average / count

            features[i, :] = sentence_average

        return features
