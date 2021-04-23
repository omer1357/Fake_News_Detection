import copy

import numpy as np
import pandas as pd
import gensim
from scipy.sparse import csr_matrix
import time


def load_data():
    fakeNews = pd.read_csv(".\Data\Fake.csv")
    trueNews = pd.read_csv(".\Data\True.csv")
    trueNews['true'] = 1
    fakeNews['true'] = 0
    newsDF = pd.concat([trueNews, fakeNews]).reset_index(drop=True)
    newsDF = newsDF.drop(['subject', 'date', 'text'], axis=1)

    '''secondDF = pd.read_csv(".\Data\Fake_Real_2nd_Data.csv")
    secondDF = secondDF.replace("REAL", 1)
    secondDF = secondDF.replace("FAKE", 0)
    secondDF = secondDF.drop(['id'], axis=1)
    secondDF = secondDF.rename(columns={"label": "true"})
    newsDF = pd.concat([newsDF, secondDF]).reset_index(drop=True)'''

    '''thirdDF = pd.read_csv(".\Data\Fake_Real_3rd_Data.csv")
    thirdDF = thirdDF.replace("Real", 1)
    thirdDF = thirdDF.replace("Fake", 0)
    thirdDF = thirdDF.rename(columns={"label": "true"})
    newsDF = pd.concat([newsDF, thirdDF]).reset_index(drop=True)'''

    forthDF = pd.read_csv(".\Data\Fake_Real_4th_Data.csv")
    forthDF = forthDF.replace("1", 0)
    forthDF = forthDF.replace("0", 1)
    forthDF = forthDF.rename(columns={"label": "true"})
    newsDF = pd.concat([newsDF, forthDF]).reset_index(drop=True)

    print("NULL values in the dataframe:\n", newsDF.isnull().sum())
    print("\n\n\nOriginal merged dataframe:\n", newsDF)

    return newsDF


def preProcess(text):
    res = []
    for word in gensim.utils.simple_preprocess(text):
        if word not in gensim.parsing.preprocessing.STOPWORDS:
            res.append(word)
    return " ".join(res)


def makeDic(texts):
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


def countVectorize(sen, dic):
    sen = sen.split()
    senVec = np.zeros(len(dic))
    for word in sen:
        if word in dic:
            senVec[dic[word]-1] += 1
    return senVec


def process_one_title(title, dic):
    title = preProcess(title)
    title = countVectorize(title, dic)
    title = csr_matrix(title)
    return title


def train_test_vectorization(df, col, train):
    start_time = time.time()
    df = df.sample(frac=1).reset_index(drop=True)
    trainSize = int(train * len(df))
    test_df = copy.deepcopy(df[trainSize:])
    df[col] = df[col].apply(preProcess)
    print("\n\n\nProcessed df:\n", df)
    trainVec = df[0:trainSize]
    dfDic = makeDic(trainVec[col].values.tolist())
    print("makeDic:")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("\n\n\nDictionary:\n", dfDic)
    df[col] = df[col].apply(countVectorize, dic=dfDic)
    print("countVector:")
    print("--- %s seconds ---" % (time.time() - start_time))
    trainVec = df[0:trainSize]
    testVec = df[trainSize:]
    print("train test split:")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Train:\n", trainVec, "\n\n\nTest:\n", testVec)
    trainX = np.stack(trainVec["title"].to_numpy())
    print("train X to stack:")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("TYPE: ", type(trainX))
    trainX = csr_matrix(trainX)
    print("train X to csr:")
    print("--- %s seconds ---" % (time.time() - start_time))
    trainY = trainVec["true"].to_numpy()
    print("Train:\n", trainVec, "\n\n\nTest:\n", testVec)
    testX = np.stack(testVec["title"].to_numpy())
    testX = csr_matrix(testX)
    print("test X csr")
    print("--- %s seconds ---" % (time.time() - start_time))
    testY = testVec["true"].to_numpy()
    return trainX, trainY, testX, testY, dfDic, test_df

