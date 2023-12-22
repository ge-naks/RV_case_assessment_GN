import numpy as np
import pandas as pd

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=.001, num_iters=1000):
        self.lr = lr
        self.num_iters = num_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        samples, features = X.shape
        self.weights = np.zeros(self.features)
        self.bias = 0

        for _ in range(self.num_iters):
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_predictions)

            dw = (1/samples) * np.dot(X.T, (predictions-y))
            db = (1/samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr.dw
            self.bias = self.bias - self.lr*db

            self
        def predict(self, X,):
            linear_predictions = np.dot(X, self.weights) + self.bias
            y_prediction = sigmoid(linear_predictions)
            class_predictions = [0 if y <= .5 else 1 for y in y_prediction]
            return class_predictions
