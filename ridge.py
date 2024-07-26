import numpy as np
import pandas as pd

def ridge_regression(X, y, alpha, learning_rate, iterations):
    n, m = X.shape
    theta = np.zeros(m)
    losses = []

    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / n) * X.T.dot(errors) + (alpha / n) * theta
        theta -= learning_rate * gradient
        loss = (1 / (2 * n)) * np.sum(errors ** 2)
        losses.append(loss)

    return theta, losses

ridge_iterations = 1000
ridge_learning_rate = 0.01
ridge_theta, ridge_losses = ridge_regression(X_train_poly_ridge, y_train, alpha_ridge, ridge_learning_rate, ridge_iterations)
y_pred_ridge = X_test_poly_ridge.dot(ridge_theta)
