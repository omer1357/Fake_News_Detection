import numpy as np
import pandas as pd

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

fakeNews = pd.read_csv("Fake.csv")
trueNews = pd.read_csv("True.csv")
trueNews['true'] = 1
fakeNews['true'] = 0
newsDF = pd.concat([trueNews, fakeNews]).reset_index(drop=True)
newsDF = newsDF.drop(['subject', 'date', 'text'], axis=1)


print("NULL values in the dataframe:\n", newsDF.isnull().sum())
print("\n\n\nOriginal dataframe after shuffle:\n", newsDF)


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


def countDicVectorize(sen, dic):
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
    return senVec[:len(sen)-rem]


# Function created to vectorize a dataframe and return it's vectorization as a list,
# But replaces with a more efficient way.
'''def vectorizeDF(df, col, classification, dic):
    vec = np.zeros(shape=(len(df), len(dic)))
    y = np.zeros(len(df))
    index = 0
    for title in df[col]:
        fake = int(df.loc[df[col] == title, classification].iloc[0])
        vec[index] = countVectorize(title, dic)
        y[index] = fake
        index += 1
    return vec, y'''


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
    #trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    print(trainX)
    print(trainX.shape)
    trainX = csr_matrix(trainX)
    trainY = trainVec["true"].to_numpy()
    testX = csr_matrix(np.stack(testVec["title"].to_numpy()))
    #testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    testX = csr_matrix(testX)
    testY = testVec["true"].to_numpy()
    return trainX, trainY, testX, testY


# Vectorize the data and split it to train/test
X_train, Y_train, X_test, Y_test = train_test_vectorization(newsDF, "title", 0.75)
print("Train:\n", X_train, "\n", Y_train, "\n\n\nTest:\n", X_test, "\n", Y_test)


'''model = Sequential()
model.add(LSTM(units=20, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=20, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=20, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=20, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=10, batch_size=64)
result = model.evaluate(X_test, Y_test)
loss = result[0]
accuracy = result[1]
print(f"[+] Accuracy: {accuracy*100:.2f}%")'''


'''newsDF["title"] = newsDF["title"].apply(preProcess)
X_train, X_test, y_train, y_test = train_test_split(newsDF.title, newsDF.true, test_size = 0.25,random_state=2)
vec_train = CountVectorizer().fit(X_train)
X_vec_train = vec_train.transform(X_train)
X_vec_test = vec_train.transform(X_test)
print(X_vec_train, "\n\n\n")
print(len(vec_train.get_feature_names()), "\n\n\n")
print(X_vec_test)'''


model = LogisticRegression(C=2)
model.fit(X_train, Y_train)
predicted_value = model.predict(X_test)
accuracy_value = roc_auc_score(Y_test, predicted_value)
print("\n\n", accuracy_value)


'''model = LogisticRegression(C=2)
model.fit(X_vec_train, y_train)
predicted_value = model.predict(X_vec_test)
accuracy_value = roc_auc_score(y_test, predicted_value)
print(accuracy_value)'''


'''model = Sequential()
model.add(LSTM(units=20, return_sequences=True))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(LSTM(units=20))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(Dense(1,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=128, callbacks=[early_stopping])
history_df = pd.DataFrame(history.history)'''



