#!/usr/bin/env python3
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

def preprocess_data(X_train , X_test): 
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

class VotedPerceptron: 
    # or SVMClassifier for svm.py
    """Classifier class.""" 
    def __init__(self, epochs=4):
        self.epochs = epochs
        self.weight_list = None
        self.count_list = None

    def train(self, X, y):
        """Fit the classifier to training data."""
        # TODO: Implement - done
        num_samples, num_features = X.shape

        current_weights = np.zeros(num_features)
        current_count = 0

        weight_list = []
        count_list = []

        for _ in range(self.epochs):
            for i in range(num_samples):
                if y[i] * (current_weights @ X[i]) <= 0:
                    if current_count > 0:
                        weight_list.append(current_weights.copy())
                        count_list.append(current_count)

                    current_weights = current_weights + y[i] * X[i]
                    current_count = 1
                else:
                    current_count += 1

        if current_count > 0:
            weight_list.append(current_weights.copy())
            count_list.append(current_count)

        self.weight_list = np.vstack(weight_list)
        self.count_list = np.array(count_list)



    def predict(self, X):
        """Predict labels for input samples.""" 
        # TODO: Implement - done
        votes = np.where((self.weight_list @ X.T) >= 0, 1, -1)
        total_vote = (self.count_list[:, None] * votes).sum(axis=0)
        predictions = np.where(total_vote >= 0, 1, -1)
        return predictions.astype(int)

def evaluate(y_true , y_pred):
    """Compute classification accuracy.""" 
    # TODO: Implement - done
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float((y_true == y_pred).mean())

def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file
: str):
    """Main function called by autograder.""" 
    # TODO: Implement - done
    # 1. Load training data
    # 2. Load test data
    # 3. Preprocess data
    # 4. Train your model
    # 5. Make predictions on test data # 6. Save predictions to pred_file
    X_train, y_train = load_data(Xtrain_file, Ytrain_file)
    X_test, _ = load_data(test_data_file)

    X_train, X_test = preprocess_data(X_train, X_test)

    model = VotedPerceptron(epochs=10)
    model.train(X_train, y_train)

    predictions = model.predict(X_test)
    pd.Series(predictions).to_csv(pred_file, index=False, header=False)