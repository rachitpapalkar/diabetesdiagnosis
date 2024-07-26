import numpy as np
import pandas as pd

learning_rate = 0.03
n_iterations = 4000

theta = np.random.randn(X_train_bias.shape[1])
n, m = X_train_bias.shape

btch_losses = []

for iteration in range(n_iterations):
    gradients = 1/n * X_train_bias.T.dot(X_train_bias.dot(theta) - y_train)
    theta = theta - learning_rate * gradients
    predictions = xi.dot(theta)
    errors = predictions - yi
    loss = (1 / (2 * n)) * np.sum(errors ** 2)
    btch_losses.append(loss)

y_train_pred_btch = X_train_bias.dot(theta)
mse_train = mean_squared_error(y_train, y_train_pred_btch)

X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_test_pred = X_test_bias.dot(theta)
mse_test = np.mean((y_test - y_test_pred) ** 2)
accu = calculate_accuracy(y_test, y_test_pred)
print("Batch Gradient Descent")
print("MSE", mse_test)
print("Accuracy", accu)
