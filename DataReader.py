import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# For now this isn't the best name overall, as we might include more datasets later on
# But for now it's a perfectly valid starting point, so we'll use it
def read_starting_dataset():
    sheet = pd.read_excel("Data/Riceetal_SupplementaryMaterials_R1.xlsx", sheet_name="SUBTL")

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

    # Each subset created will only care about those sentences were there's agreement on the definition
    # This is just the starting method, this will most definitely eliminate too much data
    # But for now it's good enough
    for word in unique_words:
        subset = sheet[sheet["probe"] == word]
        subset = subset[subset["CertainOrUncertainAgreement_R1_R2"] != 0]
        train, test = train_test_split(subset, test_size=split)

        x_test = test[["probe","line"]]
        x_train = train[["probe","line"]]

        # In this case, because we're taking it for granted that the two words have the same rating
        # We will just take the meaning given by R1
        y_test = test["Meaning_R1_BestGuess"]
        y_train = train["Meaning_R1_BestGuess"]

        x_test_set = pd.concat([x_test_set, x_test])
        x_train_set = pd.concat([x_train_set, x_train])

        y_test_set = pd.concat([y_test_set, y_test])
        y_train_set = pd.concat([y_train_set, y_train])

    return x_test_set, x_train_set, y_test_set, y_train_set