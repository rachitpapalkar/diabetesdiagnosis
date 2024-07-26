import numpy as np
import pandas as pd

def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    mse = ((y_true - y_pred) ** 2).sum() / n
    return mse

def calculate_accuracy(y_test, y_pred):
    y_pred = (y_pred >= 0.5).astype(int)
    correct_predictions = (y_pred == y_test).sum()
    total_examples = len(y_test)
    accuracy = correct_predictions / total_examples
    return accuracy * 100

train_data = pd.read_csv('standardized_training.csv')

X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
learning_rate = 0.03
n_iterations = 4000
random_state = 0
n, m = X_train_bias.shape

st_losses = []
np.random.seed(random_state)
theta = np.random.randn(X_train_bias.shape[1])

for iteration in range(n_iterations):
    random_index = np.random.randint(len(X_train_bias))
    xi = X_train_bias[random_index:random_index+1]
    yi = y_train[random_index:random_index+1]
    gradients = xi.T.dot(xi.dot(theta) - yi)
    theta = theta - learning_rate * gradients
    predictions = xi.dot(theta)
    errors = predictions - yi
    loss = (1 / (2 * n)) * np.sum(errors ** 2)
    st_losses.append(loss)

y_train_pred = X_train_bias.dot(theta)
mse_train = np.mean((y_train - y_train_pred) ** 2)

X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_test_pred = X_test_bias.dot(theta)
mse_test = np.mean((y_test - y_test_pred) ** 2)
accu = calculate_accuracy(y_test, y_test_pred)
print("Stochastic Gradient Descent")
print("MSE", mse_test)
print("Accuracy", accu)
