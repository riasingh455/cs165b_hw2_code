#!/usr/bin/env python3
"""
CMPSC 165 - Machine Learning
Homework 2, Problem 2: Support Vector Machine (SVM)
"""

import numpy as np
import pandas as pd


def load_data(X_path: str, y_path: str = None):
    """Load features and labels from CSV files."""
    # TODO: Implement - done
    X = pd.read_csv(X_path)
    X = X.values.astype(float)
    if y_path is None:
        return X, None
    y = pd.read_csv(y_path)
    y = y.values.flatten().astype(int)

    return X, y


def preprocess_data(X_train, X_test):
    """Preprocess training and test data."""
    # TODO: Implement - done
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    return X_train, X_test


class SVMClassifier:
    """Support Vector Machine Classifier."""
    def __init__(self, lambda_reg=1e-2, learning_rate=1e-2, epochs=15, shuffle=True, seed=0):
        self.lambda_reg = float(lambda_reg)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.w = None


    def train(self, X, y):
        """Fit the classifier to training data."""
        # TODO: Implement - done
        n, d = X.shape
        self.w = np.zeros(d, dtype=float)

        for _ in range(self.epochs):
            idx = self.rng.permutation(n) if self.shuffle else np.arange(n)

            for i in idx:
                xi = X[i]
                yi = y[i]
                margin = yi * (self.w @ xi)

                # From handout:
                # if yi*(w·xi) < 1:
                #   w = w + lr*(yi*xi - lambda*w)
                # else:
                #   w = w - lr*lambda*w
                if margin < 1:
                    self.w = self.w + self.learning_rate * (yi * xi - self.lambda_reg * self.w)
                else:
                    self.w = self.w - self.learning_rate * (self.lambda_reg * self.w)

        return self

    def predict(self, X):
        """Predict labels for input samples."""
        # TODO: Implement - done
        scores = X @ self.w
        return np.where(scores >= 0, 1, -1).astype(int)


def evaluate(y_true, y_pred):
    """Compute classification accuracy."""
    # TODO: Implement - done
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float((y_true == y_pred).mean())



def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file: str):
    """Main function called by autograder."""
    # TODO: Implement - done
    X_train, y_train = load_data(Xtrain_file, Ytrain_file)
    X_test, _ = load_data(test_data_file)

    X_train, X_test = preprocess_data(X_train, X_test)

    model = SVMClassifier(lambda_reg=1e-2, learning_rate=1e-2, epochs=15, shuffle=True, seed=0)
    model.train(X_train, y_train)

    predictions = model.predict(X_test)
    pd.Series(predictions).to_csv(pred_file, index=False, header=False)


# if __name__ == "__main__":
#     print("Running quick local test...")
#     run("data/spam_X.csv", "data/spam_y.csv", "data/spam_X.csv", "preds.csv")
#     print("Done. Predictions saved to preds.csv")
#     #for test local