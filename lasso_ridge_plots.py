import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lasso)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Lasso Regression: Actual vs. Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_ridge)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Ridge Regression: Actual vs. Predicted")

def plot_loss(iterations, lasso_losses, ridge_losses):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, lasso_losses, label='Lasso Loss', color='blue')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Lasso Regression: Cost vs. Iterations")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iterations, ridge_losses, label='Ridge Loss', color='red')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Ridge Regression: Cost vs. Iterations")
    plt.legend()

iterations = list(range(1, 1001))
plot_loss(iterations, lasso_losses, ridge_losses)
plt.show()
