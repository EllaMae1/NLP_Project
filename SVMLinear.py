import pandas as pd
from DataReader import split_sheet_into_test_training_per_word
import numpy as np
import spacy
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from collections import Counter

if __name__ == '__main__':

    x_test, x_train, y_test, y_train = split_sheet_into_test_training_per_word()

    words = x_train["probe"].unique()

    accuracies = []
    precisions = []
    recalls = []
    fscores = []
    classification_reports = {'1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 1},
                              '2': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 1},
                              '3': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 1},
                              '4': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 1},
                              'accuracy': 0,
                              'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 1},
                              'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 1}}

    for elem in words:

        x_train_pra = []
        x_test_pra = []
        y_train_pra = []
        y_test_pra = []
        for index, row in x_train.iterrows():
            tmp = row["probe"]
            if row["probe"] == elem:
                x_train_pra.append(row["line"])
                y_train_pra.append(y_train.loc[index][0])

        for index, row in x_test.iterrows():
            tmp = row["probe"]
            if row["probe"] == elem:
                x_test_pra.append(row["line"])
                y_test_pra.append(y_test.loc[index][0])

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
            vectorizer = TfidfVectorizer()

            X_train_tfidf = vectorizer.fit_transform(x_train_refined)
            clf = LinearSVC()
            clf.fit(X_train_tfidf, y_train_pra)

            text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                                 ('clf', LinearSVC())])

            text_clf.fit(x_train_refined, y_train_pra)
            predictions = text_clf.predict(x_test_refined)

            # print(confusion_matrix(y_test_pra, predictions))
            print(accuracy_score(y_test_pra, predictions))
            accuracies.append(accuracy_score(y_test_pra, predictions))
            # print(classification_report(y_test_pra, predictions))
            report = classification_report(y_test_pra, predictions, output_dict=True)
            print(report)
            for key in report.keys():
                if key == 'accuracy':
                    classification_reports['accuracy'] = (classification_reports['accuracy'] + report[key]) / 2
                else:
                    classification_reports[key]['precision'] = (classification_reports[key]['precision'] +
                                                                report[key]['precision']) / 2
                    classification_reports[key]['recall'] = (classification_reports[key]['recall'] +
                                                             report[key]['recall']) / 2
                    classification_reports[key]['f1-score'] = (classification_reports[key]['f1-score'] +
                                                               report[key]['f1-score']) / 2
                    classification_reports[key]['support'] = (classification_reports[key]['support'] +
                                                              report[key]['support'])

    for key in classification_reports.keys():
        if key == 'accuracy':
            classification_reports['accuracy'] = round(classification_reports['accuracy'], 2)
        else:
            classification_reports[key]['precision'] = round(classification_reports[key]['precision'], 2)
            classification_reports[key]['recall'] = round(classification_reports[key]['recall'], 2)
            classification_reports[key]['f1-score'] = round(classification_reports[key]['f1-score'], 2)
            classification_reports[key]['support'] = round(classification_reports[key]['support'], 2)
    df = pd.DataFrame(classification_reports).transpose()
    df.to_csv('report.csv')
