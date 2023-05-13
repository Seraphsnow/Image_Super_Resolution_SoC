import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_excel('data_soc.xlsx')
X = data.iloc[:, 1:4].values
y1 = data.iloc[:,-2].values
y2 = data.iloc[:,-1].values


def compute_cost(X, y, theta):
    predictions = X.dot(theta)
    cost = np.sum((predictions - y) ** 2) / (2 * len(y))
    return cost

def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1000):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        theta = theta - (learning_rate / len(X)) * X.T.dot(errors)
        cost_history[i] = compute_cost(X, y, theta)
    return theta, cost_history


X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


learning_rate = 0.1
iterations = 1000


initial_theta1 = np.zeros(3)
theta1, cost_history1 = gradient_descent(X_norm, y1, initial_theta1, learning_rate, iterations)


initial_theta2 = np.zeros(3)
theta2, cost_history2 = gradient_descent(X_norm, y2, initial_theta2, learning_rate, iterations)


plt.plot(range(1, iterations + 1), cost_history1, color='blue', label='Mangoes')
plt.plot(range(1, iterations + 1), cost_history2, color='red', label='Oranges')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost function over time')
plt.legend()
plt.show()

# Predicting the number of Mangoes and Oranges produced
new_X = np.array([30, 150, 70]) # sample input
new_X_norm = (new_X - np.mean(X, axis=0)) / np.std(X, axis=0)
mango_predictions = round(new_X_norm.dot(theta1))
orange_predictions = round(new_X_norm.dot(theta2))
print("Predicted number of Mangoes produced:", mango_predictions)
print("Predicted number of Oranges produced:", orange_predictions)