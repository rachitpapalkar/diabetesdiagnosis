import numpy as np
import pandas as pd

degree = 4
alpha_lasso = 0.01
alpha_ridge = 0.01

def create_polynomial_features(X, degree):
    X_poly = X.copy()
    for d in range(2, degree + 1):
        X_poly = np.column_stack((X_poly, X ** d))
    return X_poly

X_train_poly_lasso = create_polynomial_features(X_train, degree)
X_test_poly_lasso = create_polynomial_features(X_test, degree)
X_train_poly_ridge = create_polynomial_features(X_train, degree)
X_test_poly_ridge = create_polynomial_features(X_test, degree)

def lasso_regression(X, y, alpha, learning_rate, iterations):
    n, m = X.shape
    theta = np.zeros(m)
    losses = []

    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / n) * X.T.dot(errors) + (alpha / n) * np.sign(theta)
        theta -= learning_rate * gradient
        loss = (1 / (2 * n)) * np.sum(errors ** 2)
        losses.append(loss)

    return theta, losses

lasso_iterations = 1000
lasso_learning_rate = 0.01
lasso_theta, lasso_losses = lasso_regression(X_train_poly_lasso, y_train, alpha_lasso, lasso_learning_rate, lasso_iterations)
y_pred_lasso = X_test_poly_lasso.dot(lasso_theta)
