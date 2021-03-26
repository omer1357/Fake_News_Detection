import numpy as np


class LogisticRegressionClass:

    def __init__(self, learning_rate=0.001, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.b = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        samples_num, features_num = x.shape
        self.weights = np.zeros(features_num)
        self.b = 0
        print("start")

        for i in range(self.max_iter):
            lin = np.dot(x, self.weights) + self.b
            y_predict = self.sigmoid(lin)

            dw = (1 / samples_num) + np.dot(x.T, (y_predict - y))
            db = (1 / samples_num) + np.sum(y_predict - y)

            self.weights -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, x):
        lin = np.dot(x, self.weights) + self.b
        y_predict = self.sigmoid(lin)
        y_class = [1 if i > 0.5 else 0 for i in y_predict]

        return y_class, y_predict

