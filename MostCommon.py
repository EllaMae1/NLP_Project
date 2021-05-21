import DataReader
import pandas as pd
import numpy as np
from sklearn import metrics


class MostCommon:
    def __init__(self):
        self.x_test, self.x_train, self.y_test, self.y_train = DataReader.split_sheet_into_test_training_per_word()

    def createCounts(self):
        train_set = pd.concat([self.x_train, self.y_train], axis=1)
        del train_set['line']
        train_set.columns = ['probe', 'label']
        counts = train_set.groupby(["probe", "label"]).size().reset_index(name="count")
        most_common = counts.groupby("probe").max()
        del most_common['count']
        return most_common

    def classify(self, test_data, most_common):
        predicted = []
        for probe in test_data.iloc[:,0]:
            predicted.append(most_common['label'].loc[probe])
        return predicted


mc = MostCommon()
counts = mc.createCounts()
predicted = mc.classify(mc.x_test, counts)
testl = np.ravel(mc.y_test)
print(metrics.classification_report(testl, predicted))
