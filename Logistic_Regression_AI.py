import numpy as np
import pickle
import Data_Processor as dp


class SparseLogisticRegression:

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
            lin = x.dot(self.weights) + self.b
            y_predict = self.sigmoid(lin)

            dw = (1 / samples_num) + x.T.dot(y_predict - y)
            db = (1 / samples_num) + np.sum(y_predict - y)

            self.weights -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, x):
        lin = x.dot(self.weights) + self.b
        y_predict = self.sigmoid(lin)
        y_class = [1 if i > 0.5 else 0 for i in y_predict]

        return y_class, y_predict


def learn(x, y, x_test, y_test, dic):
    model = SparseLogisticRegression(max_iter=600)
    model.fit(x, y)
    with open("model.pkl", 'wb') as file:
        pickle.dump([model, x_test, y_test, dic], file)
    return model


def make_confusion_matrix(y_true, y_prediction):
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
    predicted_value = test_model.predict(x)[0]
    print(predicted_value)
    accuracy_value = np.sum(y == predicted_value) / len(y)
    print(make_confusion_matrix(y, predicted_value))
    return accuracy_value, make_confusion_matrix(y, predicted_value)


def load_model():
    with open("model.pkl", 'rb') as file:
        read = pickle.load(file)
    read_model = read[0]
    x_test = read[1]
    y_test = read[2]
    dic = read[3]
    return read_model, x_test, y_test, dic


def guess_one_title(title, model, dic):
    if len(title.split(" ")) < 4:
        return False, "Title is too short."
    processed_title = dp.process_one_title(title, dic)
    print(processed_title)
    if processed_title.getnnz() < 4:
        return False, "Not enough known words in the title to make a prediction."
    prediction = model.predict(processed_title)
    print(prediction)
    if int(prediction[0][0]) == 1:
        return True, "The AI thinks it's true, the prediction probability is " + "%.2f" % (prediction[1][0] * 100)
    else:
        return False, "The AI thinks it's fake, the prediction probability is " + "%.2f" % (100 - prediction[1][0] * 100)

