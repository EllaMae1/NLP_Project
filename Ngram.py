import pandas as pd
import numpy as np
from DataReader import split_sheet_into_test_training_per_word
from nltk.tokenize import regexp_tokenize
from nltk import bigrams
from nltk import FreqDist
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

def process_1(x, y, n):
    start_symbol = '<s>'
    clean = []
    for index, row in x.iterrows():
        if row['probe'] in row['line']:
            line = row['line']
            label = row['probe']+str(y.loc[index][0])
            ind= line.index(row['probe'])
            if ind-n+1>0:
                s= line[ind-n+1:ind]
                s.append(label)
            elif ind!=0:
                s = [start_symbol]*(n-1-ind)
                s.extend(line[0:ind])
                s.append(label)
            else:
                s = [start_symbol]*(n-1)
                s.append(label)
            clean.append(s)
    return clean

def process_4(x, y, n):
    start_symbol = '<s>'
    clean1, clean2, clean3, clean4 = [], [], [], []
    for index, row in x.iterrows():
        if row['probe'] in row['line']:
            line = row['line']
            label = row['probe']
            ind= line.index(row['probe'])
            if ind-n+1>0:
                s= line[ind-n+1:ind]
            elif ind!=0:
                s = [start_symbol]*(n-1-ind)
                s.extend(line[0:ind])
            else:
                s = [start_symbol]*(n-1)
            s.append(label)
            if y.loc[index][0]== '1':
                clean1.append(s)
            elif y.loc[index][0]== '2':
                clean2.append(s)
            elif y.loc[index][0]== '3':
                clean3.append(s)
            elif y.loc[index][0]== '4':
                clean4.append(s)
    return clean1, clean2, clean3, clean4

def compute_score(x_test, y_test, model,n):
    correct, error, noidea=0,0,0
    start_symbol = '<s>'
    if isinstance(model, list):
        for index, row in x_test.iterrows():
            p=[]
            line= row['line']
            if row['probe'] in line:
                ind = line.index(row['probe'])
            elif row['probe'].capitalize() in line:
                ind = line.index(row['probe'].capitalize())
            else:
                error+=1
                continue
            prior =[]
            if ind-n+1>0:
                s= line[ind-n+1:ind]
            elif ind!=0:
                s = [start_symbol]*(n-1-ind)
                s.extend(line[0:ind])
            else:
                s = [start_symbol]*(n-1)
            prior.extend(s)
            for i in range(len(model)):
                p.append(model[i].score(row['probe'],prior))
            if sum(p)<0.00001:
                noidea+=1
            elif y_test.loc[index][0].isnumeric() and p.index(max(p))+1 == (int)(y_test.loc[index][0]):
                correct+=1
            else:
                error+=1

    else:
        for index, row in x_test.iterrows():
            p=[]
            skip = False
            for i in range(4):
                line = row['line']
                if row['probe'] in line:
                    ind = line.index(row['probe'])
                elif row['probe'].capitalize() in line:
                    ind = line.index(row['probe'].capitalize())
                else:
                    skip = True
                    continue
                prior =[]
                if ind-n+1>0:
                    s= line[ind-n+1:ind]
                    prior.extend(s)
                elif ind!=0:
                    s = [start_symbol]*(n-1-ind)
                    s.extend(line[0:ind])
                    prior.extend(s)
                else:
                    s = [start_symbol]*(n-1)
                    prior.extend(s)
                p.append(model.score(row['probe']+str(i+1),prior))
            if skip:
                error += 1
                continue
            if sum(p)<0.00001:
                noidea+=1
            elif y_test.loc[index][0].isnumeric() and p.index(max(p))+1 == (int)(y_test.loc[index][0]):
                correct+=1
            else:
                error+=1
    return correct/len(x_test), error/len(x_test), noidea/len(x_test)

def get_model(x_train_clean, n):
    train_data, padded_sents = padded_everygram_pipeline(n, x_train_clean)
    model= MLE(n)
    model.fit(train_data, padded_sents)
    return model
