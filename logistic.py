import numpy as np
import pandas as pd

train_data = pd.read_csv('standardized_training.csv')
test_data = pd.read_csv('standardized_testing.csv')

X_train = train_data.drop(columns=['Outcome'])
y_train = train_data['Outcome']

X_test = test_data.drop(columns=['Outcome'])
y_test = test_data['Outcome']

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)
        X_bias = np.c_[np.ones((n_samples, 1)), X]

        for _ in range(self.n_iterations):
            linear_model = np.dot(X_bias, self.theta)
            y_predicted = self.sigmoid(linear_model)
            gradients = np.dot(X_bias.T, (y_predicted - y)) / n_samples
            self.theta -= self.learning_rate * gradients

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        n_samples, n_features = X.shape
        X_bias = np.c_[np.ones((n_samples, 1)), X]
        linear_model = np.dot(X_bias, self.theta)
        y_predicted = self.sigmoid(linear_model)
        y_predicted_labels = np.where(y_predicted >= 0.5, 1, 0)
        return y_predicted_labels

log_reg = LogisticRegressionCustom(learning_rate=0.01, n_iterations=1000)
log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)
accuracy = np.mean(predictions == y_test) * 100
print(f"Logistic Regression Accuracy: {accuracy:.2f}%")
