from DataReader import split_sheet_into_test_training_per_word
import spacy
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
import numpy as np


if __name__ == '__main__':

    model = svm.SVC()
    x_test, x_train, y_test, y_train = split_sheet_into_test_training_per_word()

    words = x_train["probe"].unique()
    accuracies = []

    for elem in words:
        x_train_pra = []
        x_test_pra = []
        y_train_pra = []
        y_test_pra = []
        for index, row in x_train.iterrows():
            tmp = row["probe"]
            if row["probe"] == elem:
                x_train_pra.append(row["line"])
                y_train_pra.append(y_train.iloc[index][0])

        for index, row in x_test.iterrows():
            tmp = row["probe"]
            if row["probe"] == elem:
                x_test_pra.append(row["line"])
                y_test_pra.append(y_test.iloc[index][0])

        nlp = spacy.load('en_core_web_sm')
        x_train_refined = []
        all_stopwords = nlp.Defaults.stop_words
        for sent in x_train_pra:
            sentence = ""
            tokens_without_sw = [word for word in sent if not word in all_stopwords]
            for item in tokens_without_sw:
                sentence = sentence + " " + item
            x_train_refined.append(sentence)

        x_test_refined = []
        for sent in x_test_pra:
            sentence = ""
            test_tokens_without_sw = [word for word in sent if not word in all_stopwords]
            for item in test_tokens_without_sw:
                sentence = sentence + " " + item
            x_test_refined.append(sentence)

        first = y_train_pra[0]
        check = False
        for j in range(1, len(y_train_pra)):
            temp = y_train_pra[j]
            if temp != first:
                check = True
                break

        if not check:
            accuracies.append(1)

        if check:
            param_grid = {'C': [0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 100, 200],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['rbf', 'poly', 'sigmoid']}
            grid = GridSearchCV(model, param_grid, refit=True)
            grid.fit(x_train_refined, y_train_pra)

            print(grid.best_params_)

            predictions = grid.predict(x_test_refined)
            print(confusion_matrix(y_test_pra, predictions))
            print(accuracy_score(y_test_pra, predictions))

    print(np.mean(accuracies))