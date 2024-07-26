import numpy as np
import pandas as pd

class LeastSquaresClassifier:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_bias = np.c_[np.ones((n_samples, 1)), X]
        self.theta = np.linalg.pinv(X_bias).dot(y)

    def predict(self, X):
        n_samples, n_features = X.shape
        X_bias = np.c_[np.ones((n_samples, 1)), X]
        linear_model = np.dot(X_bias, self.theta)
        y_predicted_labels = np.where(linear_model >= 0.5, 1, 0)
        return y_predicted_labels

least_squares_clf = LeastSquaresClassifier()
least_squares_clf.fit(X_train, y_train)
predictions = least_squares_clf.predict(X_test)
accuracy = np.mean(predictions == y_test) * 100
print(f"Least Squares Classifier Accuracy: {accuracy:.2f}%")
