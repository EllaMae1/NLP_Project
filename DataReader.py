import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from sklearn.linear_model import LogisticRegression


# For now this isn't the best name overall, as we might include more datasets later on
# But for now it's a perfectly valid starting point, so we'll use it
def read_starting_dataset():
    sheet = pd.read_excel("Data/Riceetal_SupplementaryMaterials_R1.xlsx", sheet_name="SUBTL", engine='openpyxl')

    return sheet


"""Idea of this method is that for each unique word, it'll find a test train split for it
This created split is what will be used later for the training"""


# Split is how many are used for training, and how many are used for testing
# Where the percentage less than split represents the test values
# And the percentage greater represents the training values
def split_sheet_into_test_training_per_word(sheet=None, split=0.25):
    if sheet is None:
        sheet = read_starting_dataset()

    unique_words = sheet["probe"].unique()

    x_test_set = pd.DataFrame()
    x_train_set = pd.DataFrame()
    y_train_set = pd.DataFrame()
    y_test_set = pd.DataFrame()

    # Forgot to tokenize beforehand, doing it now
    sheet["line"] = sheet.apply(lambda row: nltk.word_tokenize(row["line"]), axis=1)

    # Each subset created will only care about those sentences were there's agreement on the definition
    # This is just the starting method, this will most definitely eliminate too much data
    # But for now it's good enough
    for word in unique_words:
        subset = sheet[sheet["probe"] == word]
        subset = subset[subset["CertainOrUncertainAgreement_R1_R2"] != 0]

        subset = subset[subset["Meaning_R1_BestGuess"] != "-"]
        subset = subset[subset["Meaning_R2_BestGuess"] != "-"]

        unique_definitions = subset["Meaning_R1_BestGuess"].unique()
        if len(unique_definitions) < 2:
            continue

        train, test = train_test_split(subset, test_size=split)

        x_test = test[["probe", "line"]]
        x_train = train[["probe", "line"]]

        # In this case, because we're taking it for granted that the two words have the same rating
        # We will just take the meaning given by R1


        y_test = test["Meaning_R1_BestGuess"]
        y_train = train["Meaning_R1_BestGuess"]

        x_test_set = pd.concat([x_test_set, x_test])
        x_train_set = pd.concat([x_train_set, x_train])

        y_test_set = pd.concat([y_test_set, y_test])
        y_train_set = pd.concat([y_train_set, y_train])

    y_test_set = y_test_set.astype("str")
    y_train_set = y_train_set.astype("str")
    return x_test_set, x_train_set, y_test_set, y_train_set


# The difference in this method vs default is that we no longer train on the original word
# We train on word1, word2, so it's a slightly modified version of the word
# There are definitely arguments that this might be too nice, but it can still be used

# The idea of simplified is that if true it'll return the sentence as x
# and then the word that needs to be predicted as y
# This needs to be modified somehow depending on what approach is gonna be taken
# For example sometimes it might be worth it to hide what the correct word is
# But screw it, this works
# One thing of note is that this makes it so that the test set has the correct answer in the sentence
# Might change it later to work properly
def create_test_training_data_with_modified_words(sheet=None, split=0.25, simplified=True):
    if sheet is None:
        sheet = read_starting_dataset()

    x_test_set, x_train_set, y_test_set, y_train_set = split_sheet_into_test_training_per_word(sheet=sheet, split=split)

    x_test_set["probe"] = x_test_set["probe"] + y_test_set.to_string()
    x_train_set["probe"] = x_train_set["probe"] + y_train_set.to_string()

    for row in x_test_set.iterrows():
        sentence = row["line"]
        # I just want to ignore the last character
        mod_word = row["probe"]
        orig_word = mod_word[:-1]

        # The only way I can think of is by going through each word in the sentence
        for word in sentence:
            if word == orig_word:
                # I'm not completely sure if this'll modify the original version as well
                # I'll bug test this later
                word = mod_word

    for row in x_train_set.iterrows():
        sentence = row["line"]
        # I just want to ignore the last character
        mod_word = row["probe"]
        orig_word = mod_word[:-1]

        # The only way I can think of is by going through each word in the sentence
        for word in sentence:
            if word == orig_word:
                word = mod_word

    # What this block does in a nutshell is just make it so that the test set are the sentences with the homogram
    # And the y set are the word that's being looked for
    # As mentioned in the start, this makes it so that the test and train set contain the actual word
    # Which is kinda shitty, but how to fix it depends greatly on what approach will be done
    if simplified:
        y_test_set = x_test_set["probe"]
        y_train_set = x_train_set["probe"]
        x_test_set = x_test_set["line"]
        x_train_set = x_train_set["line"]

    return x_test_set, x_train_set, y_test_set, y_train_set


if __name__ == "__main__":
    x_test_set, x_train_set, y_test_set, y_train_set = split_sheet_into_test_training_per_word()
    print(y_test_set.drop_duplicates())
    print(y_train_set.drop_duplicates())
