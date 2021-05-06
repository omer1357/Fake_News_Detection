"""
This file is responsible of all the functions that uses the data.
From loading it to preprocessing and vectorizing it.
"""

# Imports
import copy
import numpy as np
import pandas as pd
import gensim
from scipy.sparse import csr_matrix


def load_data():  # Function to load the 2 datasets and combine them.
    fakeNews = pd.read_csv(".\Data\Fake.csv")
    trueNews = pd.read_csv(".\Data\True.csv")
    trueNews['true'] = 1
    fakeNews['true'] = 0
    newsDF = pd.concat([trueNews, fakeNews]).reset_index(drop=True)
    newsDF = newsDF.drop(['subject', 'date', 'text'], axis=1)

    secDF = pd.read_csv(".\Data\Fake_Real_2nd_Data.csv")
    secDF = secDF.replace("1", 0)
    secDF = secDF.replace("0", 1)
    secDF = secDF.rename(columns={"label": "true"})
    newsDF = pd.concat([newsDF, secDF]).reset_index(drop=True)

    print("----Data Loaded----\n\n")
    return newsDF


def preProcess(text):  # Function to preprocess the data - remove stop words and special characters, lowercase all, etc.
    res = []
    for word in gensim.utils.simple_preprocess(text):
        if word not in gensim.parsing.preprocessing.STOPWORDS:
            res.append(word)
    return " ".join(res)


def makeDic(texts):  # Function to make a dictionary containing all the words in a given texts array.
    words = []
    dic = {}
    cnt = 1
    for sen in texts:
        for word in sen.split():
            if word not in words:
                if word.encode().isalpha():
                    words.append(word)
    words.sort()
    for word in words:
        if word not in dic:
            dic[word] = cnt
            cnt += 1
    return dic


def countVectorize(sen, dic):  # Function to count vectorize a given sentence using a given dictionary
    sen = sen.split()
    senVec = np.zeros(len(dic))
    for word in sen:
        if word in dic:
            senVec[dic[word]-1] += 1
    return senVec


def process_one_title(title, dic):
    """
    Function to full process a single title (from preprocess to count vectorization and sparse matrix representation).
    """
    title = preProcess(title)
    title = countVectorize(title, dic)
    title = csr_matrix(title)
    return title


def train_test_vectorization(df, col, train):
    """
    Function to full process a given pandas dataframe
    (from preprocess to count vectorization and sparse matrix representation)
    As well as splitting to train, test and x, y by a given train/test rate.
    """

    df = df.sample(frac=1).reset_index(drop=True)
    trainSize = int(train * len(df))
    test_df = copy.deepcopy(df[trainSize:])
    df[col] = df[col].apply(preProcess)
    print("----Preprocess Completed----\n\n")

    trainVec = df[0:trainSize]
    dfDic = makeDic(trainVec[col].values.tolist())
    df[col] = df[col].apply(countVectorize, dic=dfDic)
    print("----Data Vectorized----\n\n")

    trainVec = df[0:trainSize]
    testVec = df[trainSize:]
    trainX = np.stack(trainVec["title"].to_numpy())
    trainX = csr_matrix(trainX)
    trainY = trainVec["true"].to_numpy()
    testX = np.stack(testVec["title"].to_numpy())
    testX = csr_matrix(testX)
    testY = testVec["true"].to_numpy()
    print("----Data Handled----\n\n")
    return trainX, trainY, testX, testY, dfDic, test_df

