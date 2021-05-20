from DataReader import split_sheet_into_test_training_per_word
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

    for index, row in x_test.iterrows():
        if row["probe"] == x_test["probe"].unique()[0]:
            x_test_pra.append(row["line"])
            y_test_pra.append(y_test[0][index])
        else:
            break

    nlp = spacy.load('en_core_web_sm')
    x_train_refined = []
    all_stopwords = nlp.Defaults.stop_words
    for index,sent in x_train_pra.iterrows():
        text_tokens = nlp(sent)
        #     text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
        x_train_refined.append(tokens_without_sw)

    x_test_refined = []
    for index,sent in x_test_pra.iterrows():
        test_tokens = nlp(sent)
        test_tokens_without_sw = [word for word in test_tokens if not word in all_stopwords]
        x_test_refined.append(test_tokens_without_sw)


    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(x_train_refined)
    clf = LinearSVC()
    clf.fit(X_train_tfidf, y_train)

    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', LinearSVC())])

    text_clf.fit(x_train_refined, y_train_pra)
    predictions = text_clf.predict(x_test_refined)

    print(confusion_matrix(y_test_pra, predictions))
    print(accuracy_score(y_test_pra, predictions))
    print(classification_report(y_test_pra, predictions))