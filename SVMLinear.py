from DataReader import split_sheet_into_test_training_per_word
import numpy as np
import spacy
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

if __name__ == '__main__':

    x_test, x_train, y_test, y_train = split_sheet_into_test_training_per_word()

    words = x_train["probe"].unique()
    accuracies = []
    i = 0
    while i < len(words):
        x_train_pra = []
        x_test_pra = []
        y_train_pra = []
        y_test_pra = []
        for index, row in x_train.iterrows():
            if row["probe"] == x_train["probe"].unique()[i]:
                x_train_pra.append(row["line"])
                y_train_pra.append(y_train[0][index])
            else:
                break

        for index, row in x_test.iterrows():
            if row["probe"] == x_test["probe"].unique()[i]:
                x_test_pra.append(row["line"])
                y_test_pra.append(y_test[0][index])
            else:
                break

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

        print(y_train_pra)
        first = y_train_pra[0]
        # first = '1'
        check = False
        for j in range(1, len(y_train_pra)):
            if y_train_pra[j] != first:
                check = True

        # if not check:
        #     accuracies.append(1)
        print(check)
        print(accuracies)
        if check:
            vectorizer = TfidfVectorizer()

            X_train_tfidf = vectorizer.fit_transform(x_train_refined)
            clf = LinearSVC()
            clf.fit(X_train_tfidf, y_train_pra)

            text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                             ('clf', LinearSVC())])

            text_clf.fit(x_train_refined, y_train_pra)
            predictions = text_clf.predict(x_test_refined)

            print(confusion_matrix(y_test_pra, predictions))
            print(accuracy_score(y_test_pra, predictions))
            accuracies.append(accuracy_score(y_test_pra, predictions))
            print(classification_report(y_test_pra, predictions))
        i += 1

    print(np.mean(accuracies))
