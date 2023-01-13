"""Logistic regression model."""


import numpy as np



class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """

        self.w = 0  # TODO: change this
        self.b = 0
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5


    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        s = 1 / (1 + np.exp(-z))
        return s

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        # TODO: implement me
        y_train = y_train.T
        X_train = X_train.T
        self.w = np.zeros((X_train.shape[0], 1))
        #pred = X_test.dot(self.w).argmax(axis=1)
        for i in range(self.epochs):
            m = X_train.shape[1]
            P = self.sigmoid(np.dot(self.w.T, X_train) + self.b)
            dz = (P - y_train)
            dw = np.dot(X_train, dz.T) / m
            db = np.sum(dz) / m
            grads = {"dw": dw, "db": db}

            dw_ = grads["dw"]  # get dw
            db_ = grads["db"]  # get db

            #update weight and base
            self.w = self.w - self.lr * dw_
            self.b = self.b - self.lr * db_

        pass


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test = X_test.T
        # number of examples
        num = X_test.shape[1]
        labels = np.zeros((1, num))
        #weight is the first array in ndarray of X
        self.w = self.w.reshape(X_test.shape[0], 1)

        # sig function probability
        sig_prob = self.sigmoid(np.dot(self.w.T, X_test) + self.b)

        #np[a,b] = values from a before b
        for i in range(sig_prob.shape[1]):
            #if sig_prob[0,i] > 0.5 true, then 1, else 0
            labels[0, i] = np.where(sig_prob[0, i] > 0.5, 1, 0)
            pass

        return labels




