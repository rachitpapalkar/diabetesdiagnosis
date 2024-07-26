import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('standardized_training.csv')
test_data = pd.read_csv('standardized_testing.csv')

X_train = train_data.drop(columns=['Outcome'])
y_train = train_data['Outcome']

X_test = test_data.drop(columns=['Outcome'])
y_test = test_data['Outcome']

gini_tree = DecisionTreeClassifier(criterion='gini')
gini_tree.fit(X_train, y_train)
gini_predictions = gini_tree.predict(X_test)
gini_accuracy = accuracy_score(y_test, gini_predictions)
print(f"Gini Decision Tree Accuracy: {gini_accuracy * 100:.2f}%")

entropy_tree = DecisionTreeClassifier(criterion='entropy')
entropy_tree.fit(X_train, y_train)
entropy_predictions = entropy_tree.predict(X_test)
entropy_accuracy = accuracy_score(y_test, entropy_predictions)
print(f"Entropy Decision Tree Accuracy: {entropy_accuracy * 100:.2f}%")
