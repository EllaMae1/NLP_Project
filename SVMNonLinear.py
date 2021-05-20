from DataReader import split_sheet_into_test_training_per_word
import spacy
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import GridSearchCV

# class SVM:
#
#     def __init__(self):
#         self.model = svm.SVC()
#
#     def createFeatures(self):
#
#         x_test, x_train, y_test, y_train = split_sheet_into_test_training_per_word()
#
#         x_train_pra = []
#         x_test_pra = []
#         for index, row in x_train.iterrows():
#             if row["probe"] == x_train["probe"].unique()[0]:
#                 x_train_pra.append(row["line"])
#             else:
#                 break
#
#         for word in x_test["probe"]:
#             if word == x_test["probe"].unique()[0]:
#                 x_test_pra.append(x_test["line"])
#
#         for index, row in x_test.iterrows():
#             if row["probe"] == x_test["probe"].unique()[0]:
#                 x_test_pra.append(row["line"])
#             else:
#                 break
#
#         nlp = spacy.load('en_core_web_lg')
#         x_train_refined = []
#         all_stopwords = nlp.Defaults.stop_words
#         for sent in x_train_pra:
#             text_tokens = nlp(sent)
#             #     text_tokens = word_tokenize(text)
#             tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
#             x_train_refined.append(tokens_without_sw)
#
#
#
#         param_grid = {'C': [0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 100, 200],
#                            'gamma': [1, 0.1, 0.01, 0.001],
#                            'kernel': ['rbf', 'poly', 'sigmoid']}
#         grid = GridSearchCV(model, param_grid, refit=True)
#         grid.fit(x_train_refined, y_train)
#
#         print(grid.best_params_)
#
#         predictions = grid.predict(feature_test)
#         print(confusion_matrix(target_test, predictions))
#         print(accuracy_score(target_test, predictions))

if __name__ == '__main__':

    model = svm.SVC()
    x_test, x_train, y_test, y_train = split_sheet_into_test_training_per_word()

    x_train_pra = []
    x_test_pra = []
    y_train_pra = []
    y_test_pra = []
    for index, row in x_train.iterrows():
        if row["probe"] == x_train["probe"].unique()[0]:
            x_train_pra.append(row["line"])
            y_train_pra.append(y_train[0][index])
        else:
            break

    # for word in x_test["probe"]:
    #     if word == x_test["probe"].unique()[0]:
    #         x_test_pra.append(x_test["line"])

    for index, row in x_test.iterrows():
        if row["probe"] == x_test["probe"].unique()[0]:
            x_test_pra.append(row["line"])
            y_test_pra.append(y_test[0][index])
        else:
            break

    nlp = spacy.load('en_core_web_lg')
    x_train_refined = []
    all_stopwords = nlp.Defaults.stop_words
    for sent in x_train_pra:
        text_tokens = nlp(sent)
        #     text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
        x_train_refined.append(tokens_without_sw)

    x_test_refined = []
    for sent in x_test_pra:
        test_tokens = nlp(sent)
        test_tokens_without_sw = [word for word in test_tokens if not word in all_stopwords]
        x_test_refined.append(test_tokens_without_sw)

    param_grid = {'C': [0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 100, 200],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['rbf', 'poly', 'sigmoid']}
    grid = GridSearchCV(model, param_grid, refit=True)
    grid.fit(x_train_refined, y_train_pra)

    print(grid.best_params_)

    predictions = grid.predict(x_test_refined)
    print(confusion_matrix(y_test_pra, predictions))
    print(accuracy_score(y_test_pra, predictions))