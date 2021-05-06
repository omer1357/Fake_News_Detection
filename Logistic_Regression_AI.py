"""
This file contains the logistoc regression AI and all the functions using it.
Including the training, testing, making the confusion matrix, etc.
"""

# Imports
import numpy as np
import pickle
import Data_Processor as dp


class SparseLogisticRegression:
    """
    Class that represents a logistic regression AI that can be trained, tested and used.
    """

    def __init__(self, learning_rate=0.001, max_iter=1000):  # initialization
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.b = None

    def sigmoid(self, x):
        """
        Function to return the result of sigmoid function on a given value (makes the value between 0 to 1).
        """
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):  # Function to train the logistic regression algorithm.
        samples_num, features_num = x.shape
        self.weights = np.zeros(features_num)
        self.b = 0
        print("----AI Training Started----\n\n")

        for i in range(self.max_iter):
            lin = x.dot(self.weights) + self.b
            y_predict = self.sigmoid(lin)

            dw = (1 / samples_num) + x.T.dot(y_predict - y)
            db = (1 / samples_num) + np.sum(y_predict - y)

            self.weights -= self.learning_rate * dw
            self.b -= self.learning_rate * db
        print("----AI Training Started----\n\n")

    def predict(self, x):  # Function to make a prediction based on the class trained algorithm.
        lin = x.dot(self.weights) + self.b
        y_predict = self.sigmoid(lin)
        y_class = [1 if i > 0.5 else 0 for i in y_predict]

        return y_class, y_predict


def learn(x, y, x_test, y_test, dic, test_pd):
    """
    Function that creates a logistic regression class and train it.
    After the training, the function saves the trained algorithm as well as few needed variables.
    """
    model = SparseLogisticRegression(max_iter=600)
    model.fit(x, y)
    with open("model.pkl", 'wb') as file:
        pickle.dump([model, x_test, y_test, dic, test_pd], file)
    return model


def make_confusion_matrix(y_true, y_prediction):  # Function to calculate a confusion matrix.
    tn, fn, fp, tp = 0, 0, 0, 0
    for i in range(len(y_prediction)):
        if y_prediction[i] == y_true[i]:
            if y_prediction[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y_prediction[i] == 1:
                fp += 1
            else:
                fn += 1
    return [[tn, fp], [fn, tp]]


def test(test_model, x, y):
    """
    Function that tests a trained model on it's training data.
    Then returns it's accuracy as well as confusion matrix.
    """
    predicted_value = test_model.predict(x)[0]
    accuracy_value = np.sum(y == predicted_value) / len(y)
    return accuracy_value, make_confusion_matrix(y, predicted_value)


def load_model():  # Function to load a previously saved model and few needed variables.
    with open("model.pkl", 'rb') as file:
        read = pickle.load(file)
    read_model = read[0]
    x_test = read[1]
    y_test = read[2]
    dic = read[3]
    test_pd = read[4]
    return read_model, x_test, y_test, dic, test_pd


def guess_one_title(title, model, dic):
    """
    Function that uses a given model (Logistic Regression class object) to predict a given title.
    """
    if len(title.split(" ")) < 4:
        return False, "Title is too short."
    processed_title = dp.process_one_title(title, dic)
    if processed_title.getnnz() < 4:
        return False, "Not enough known words in the title to make a prediction."
    prediction = model.predict(processed_title)
    if int(prediction[0][0]) == 1:
        return True, "The AI thinks it's true, the prediction probability is " + "%.2f" % (prediction[1][0] * 100)
    else:
        return False, "The AI thinks it's fake, the prediction probability is " + "%.2f" % (100 - prediction[1][0] * 100)

