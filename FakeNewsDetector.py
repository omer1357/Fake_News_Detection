import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import gensim
from scipy.sparse import csr_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pickle


def load_data():
    fakeNews = pd.read_csv("Fake.csv")
    trueNews = pd.read_csv("True.csv")
    trueNews['true'] = 1
    fakeNews['true'] = 0
    newsDF = pd.concat([trueNews, fakeNews]).reset_index(drop=True)
    newsDF = newsDF.drop(['subject', 'date', 'text'], axis=1)

    secondDF = pd.read_csv("Fake_Real_2nd_Data.csv")
    secondDF = secondDF.replace("REAL", 1)
    secondDF = secondDF.replace("FAKE", 0)
    secondDF = secondDF.drop(['id'], axis=1)
    secondDF = secondDF.rename(columns={"label": "true"})
    newsDF = pd.concat([newsDF, secondDF]).reset_index(drop=True)

    thirdDF = pd.read_csv("Fake_Real_3rd_Data.csv")
    thirdDF = thirdDF.replace("Real", 1)
    thirdDF = thirdDF.replace("Fake", 0)
    thirdDF = thirdDF.rename(columns={"label": "true"})
    newsDF = pd.concat([newsDF, thirdDF]).reset_index(drop=True)

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


'''def countDicVectorize(sen, dic):
    sen = sen.split()
    senVec = np.zeros(len(sen))
    cnt = 0
    rem = 0
    for word in sen:
        if word in dic:
            senVec[cnt] += dic[word]
            cnt += 1
        else:
            rem += 1
    return senVec[:len(sen)-rem]'''


def get_len_data(col):
    lens = []
    for i in col:
        lens.append(len(i))
    lenpd = pd.DataFrame(lens, columns=["len"])
    return lenpd.describe(), int(lenpd.max())


def set_len(arr, target):
    if len(arr) > target:
        arr = arr[0:target]
    if len(arr) != target:
        need = target - len(arr)
        arr = np.insert(arr, len(arr), [0]*need)
    return arr


def train_test_vectorization(df, col, train):
    df = df.sample(frac=1).reset_index(drop=True)
    df[col] = df[col].apply(preProcess)
    print("\n\n\nProcessed df:\n", df)
    trainSize = int(train * len(df))
    trainVec = df[0:trainSize]
    dfDic = makeDic(trainVec[col].values.tolist())
    print("\n\n\nDictionary:\n", dfDic)
    df[col] = df[col].apply(countVectorize, dic=dfDic)
    lenDet, maxL = get_len_data(df[col])
    print("\n\nData length details:\n", lenDet, "\n\n")
    trainVec = df[0:trainSize]
    testVec = df[trainSize:]
    print("Train:\n", trainVec, "\n\n\nTest:\n", testVec)
    trainX = np.stack(trainVec["title"].to_numpy())
    print(trainX)
    print(trainX.shape)
    trainX = csr_matrix(trainX)
    trainY = trainVec["true"].to_numpy()
    testX = np.stack(testVec["title"].to_numpy())
    testX = csr_matrix(testX)
    testY = testVec["true"].to_numpy()
    return trainX, trainY, testX, testY, dfDic


def process_title(title, dic):
    title = preProcess(title)
    title = countVectorize(title, dic)
    title = csr_matrix(title)
    return title


def learn(x, y, x_test, y_test, dic):
    learned_model = LogisticRegression(C=2)
    learned_model.fit(x, y)
    with open("model.pkl", 'wb') as file:
        pickle.dump([learned_model, x_test, y_test, dic], file)
    return learned_model


def load_model():
    with open("model.pkl", 'rb') as file:
        read = pickle.load(file)
    read_model = read[0]
    x_test = read[1]
    y_test = read[2]
    dic = read[3]
    return read_model, x_test, y_test, dic


def test(test_model, x, y):
    predicted_value = test_model.predict(x)
    accuracy_value = roc_auc_score(y, predicted_value)
    return accuracy_value, confusion_matrix(y, predicted_value)


def show_cf(cf):
    group_labels = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf, annot=labels, fmt="", cmap='Blues')
    plt.show()


def start_program():
    df = load_data()
    relearn = input("Would you like the ai to relearn the data? (Y/N)\n")
    if relearn == "Y":
        X_train, Y_train, X_test, Y_test, dic = train_test_vectorization(df, "title", 0.75)
        model = learn(X_train, Y_train, X_test, Y_test, dic)
        acc, cf = test(model, X_test, Y_test)
        print("Relearn process completed:", acc, "Success rate.")
    else:
        model, X_test, Y_test, dic = load_model()
        acc, cf = test(model, X_test, Y_test)
        print("Model loaded.", acc, "Success rate.")

    present_cf = input("Would you like to see the ai confusion matrix? (Y/N)\n")
    if present_cf == "Y":
        show_cf(cf)

    title = str(input("Enter title (enter 0 to stop the program)\n"))
    while title != "0":
        processed_title = process_title(title, dic)
        print(model.predict(processed_title))
        title = str(input("Enter title (enter 0 to stop the program)\n"))


start_program()

