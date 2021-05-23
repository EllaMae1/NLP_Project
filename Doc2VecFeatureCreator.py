import numpy as np
import pandas as pd
import gensim
import gensim.models
import gensim.downloader
import gensim.test.test_data
import os
import smart_open


class Doc2VecFeatureCreator:
    def __init__(self):
        # This feature creator is super lazy and just uses a default method, but it's here!

        # I'm also not making it create the features on itself,
        # If I did, then I'd be predicting words based off training data, so there wouldn't be much learning going on
        test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
        lee_train_file = os.path.join(test_data_dir, 'lee_background.cor')
        data = list(read_corpus(lee_train_file))

        # Setting a minimum count of 2 is apparently always helpful, so i'm just following instructions
        self.model = gensim.models.doc2vec.Doc2Vec(min_count=2, epochs=25)
        self.model.build_vocab(data)
        self.model.train(data, total_examples=self.model.corpus_count, epochs=self.model.epochs)


    def create_feature_set(self, data):
        vector_length = self.model.vector_size
        features = np.zeros((len(data), vector_length))
        for i in range(len(data)):
            sentence = data.iloc[i]["line"]

            features[i, :] = self.model.infer_vector(sentence)

        return features


# These lines of code to process the existing corpora were followed from the gensim documentation
# They were modified to better suit our needs, but we got the idea on how to do it from them
# Here is the source: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py
def read_corpus(fname):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, sentence in enumerate(f):
            tokens = gensim.utils.simple_preprocess(sentence)
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
