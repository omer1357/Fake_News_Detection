import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pygame
import sys

import gensim
from scipy.sparse import csr_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pickle

import tkinter as tk

from Logistic_Regression import LogisticRegressionClass
import Graphic_Handler as gh


def load_data():
    fakeNews = pd.read_csv("Fake.csv")
    trueNews = pd.read_csv("True.csv")
    trueNews['true'] = 1
    fakeNews['true'] = 0
    newsDF = pd.concat([trueNews, fakeNews]).reset_index(drop=True)
    newsDF = newsDF.drop(['subject', 'date', 'text'], axis=1)

    '''secondDF = pd.read_csv("Fake_Real_2nd_Data.csv")
    secondDF = secondDF.replace("REAL", 1)
    secondDF = secondDF.replace("FAKE", 0)
    secondDF = secondDF.drop(['id'], axis=1)
    secondDF = secondDF.rename(columns={"label": "true"})
    newsDF = pd.concat([newsDF, secondDF]).reset_index(drop=True)

    thirdDF = pd.read_csv("Fake_Real_3rd_Data.csv")
    thirdDF = thirdDF.replace("Real", 1)
    thirdDF = thirdDF.replace("Fake", 0)
    thirdDF = thirdDF.rename(columns={"label": "true"})
    newsDF = pd.concat([newsDF, thirdDF]).reset_index(drop=True)'''

    forthDF = pd.read_csv("Fake_Real_4th_Data.csv")
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


'''ef countDicVectorize(sen, dic):
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


'''def set_len(arr, target):
    if len(arr) > target:
        arr = arr[0:target]
    if len(arr) != target:
        need = target - len(arr)
        arr = np.insert(arr, len(arr), [0]*need)
    return arr'''


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


'''def train_test_dicvectorization(df, col, train):
    df = df.sample(frac=1).reset_index(drop=True)
    df[col] = df[col].apply(preProcess)
    print("\n\n\nProcessed df:\n", df)
    trainSize = int(train * len(df))
    trainVec = df[0:trainSize]
    dfDic = makeDic(trainVec[col].values.tolist())
    print("\n\n\nDictionary:\n", dfDic)
    df[col] = df[col].apply(countDicVectorize, dic=dfDic)
    lenDet, maxL = get_len_data(df[col])
    df[col] = df[col].apply(set_len, target=20)
    print("\n\nData length details:\n", lenDet, "\n\n")
    trainVec = df[0:trainSize]
    testVec = df[trainSize:]
    print("Train:\n", trainVec, "\n\n\nTest:\n", testVec)
    trainX = np.stack(trainVec["title"].to_numpy())
    print(trainX)
    print(trainX.shape)
    trainY = trainVec["true"].to_numpy()
    testX = np.stack(testVec["title"].to_numpy())
    testY = testVec["true"].to_numpy()
    return trainX, trainY, testX, testY, dfDic'''


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


def guess(title, model, dic):
    if len(title.split(" ")) < 4:
        print("Title is too short.")
        return False
    processed_title = process_title(title, dic)
    if processed_title.getnnz() < 4:
        print("Not enough known words in the title to make a prediction.")
        return False
    prediction = model.predict(processed_title)
    if int(prediction[0]) == 1:
        return True, "%.2f" % (model.predict_proba(processed_title)[0][1]*100)
    else:
        return False, "%.2f" % (model.predict_proba(processed_title)[0][0]*100)


def check_quit(event):
    if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()


def wait_yes_no():
    while True:
        for event in pygame.event.get():
            check_quit(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if 760 < event.pos[1] < 855:
                    if 80 < event.pos[0] < 220:
                        return True
                    elif 380 < event.pos[0] < 795:
                        return False


def wait_text_input(screen, yt, yb, font_size, img):
    input_box = pygame.Rect(40, yt, 505, 90)
    color_inactive = (0, 0, 0)
    color_active = (194, 0, 0)
    color = color_inactive
    active = False
    text = ''
    multiline = []
    font = pygame.font.Font(None, font_size)

    while True:
        for event in pygame.event.get():
            check_quit(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                # If the user clicked on the input_box rect.
                if input_box.collidepoint(event.pos):
                    # Toggle the active variable.
                    active = not active
                else:
                    active = False
                # Change the current color of the input box.
                color = color_active if active else color_inactive

                if 220 < event.pos[0] < 380 and yb < event.pos[1] < yb + 70:
                    return "".join(multiline) + text

            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        return "".join(multiline) + text
                    elif event.key == pygame.K_BACKSPACE:
                        if text == "":
                            text = multiline[-1]
                            multiline = multiline[:-1]
                        text = text[:-1]
                    else:
                        text += event.unicode

        # Render the current text.
        gh.set_screen(screen, img)
        txt_surface = font.render(text, True, color)
        if txt_surface.get_width() + 25 > 505:
            multiline.append(text)
            text = ''
        if multiline:
            line = 0
            multiline.append(text)
            for i in multiline:
                txt_surface = font.render(i, True, color)
                screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5 + line*25))
                line += 1
            multiline = multiline[:-1]
        else:
            screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))

        pygame.draw.rect(screen, color, input_box, 2)
        pygame.display.flip()


def start_program():
    df = load_data()
    screen = gh.init_pygame()
    gh.set_screen(screen, 0)

    relearn = wait_yes_no()
    if relearn:
        gh.set_screen(screen, 3)
        train_size = 0.8
        X_train, Y_train, X_test, Y_test, dic = train_test_vectorization(df, "title", train_size)
        model = learn(X_train, Y_train, X_test, Y_test, dic)
        acc, cf = test(model, X_test, Y_test)
        gh.set_screen(screen, 2)
        print("Relearn process completed:", acc, "Success rate.")
    else:
        model, X_test, Y_test, dic = load_model()
        acc, cf = test(model, X_test, Y_test)
        gh.set_screen(screen, 1)
        print("Model loaded.", acc, "Success rate.")

    '''X_train, Y_train, X_test, Y_test, dic = train_test_vectorization(df, "title", 0.8)
    mod = LogisticRegressionClass(max_iter=100)
    mod.fit(X_train, Y_train)
    pred = mod.predict(X_test)[0]
    print("acc: ", np.sum(Y_test == pred)/len(Y_test))'''

    present_cf = wait_yes_no()
    if present_cf:
        show_cf(cf)

    gh.set_screen(screen, 4)
    title = wait_text_input(screen, 655, 790, 35, 4)
    print(title)
    while title != "0":
        predict = guess(title, model, dic)
        print(predict)
        if predict:
            gh.set_screen(screen, 6)
            title = wait_text_input(screen, 693, 810, 35, 6)
            print(title)
        else:
            gh.set_screen(screen, 5)
            title = wait_text_input(screen, 693, 810, 35, 5)
            print(title)


start_program()
