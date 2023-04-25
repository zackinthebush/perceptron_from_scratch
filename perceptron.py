import random
import numpy as np
import matplotlib.pyplot as plt
import csv

# Génération des données d'apprentissage
def generate_data(n):
    X = np.zeros((n, 2))
    y = np.zeros(n)
    for i in range(n):
        X[i, 0] = random.uniform(0, 1)
        X[i, 1] = random.uniform(0, 1)
        if X[i, 0] + X[i, 1] - 1 > 0:
            y[i] = 1
        else:
            y[i] = -1
    return X, y

# Définition de la fonction seuil
def threshold(x):
    if x > 0:
        return 1
    else:
        return -1

# Définition de la classe Neurone
class Neurone:
    def __init__(self):
        self.weights = np.random.uniform(-1, 1, 2)
        self.bias = 0.5
        self.output = 0

    def compute_output(self, x):
        s = np.dot(self.weights, x) + self.bias
        self.output = threshold(s)
        return self.output

    def update_weights(self, x, error, learning_rate):
        self.bias += learning_rate * error
        self.weights += learning_rate * error * x

# Initialisation du perceptron
def init_perceptron():
    return Neurone()

# Entraînement du perceptron
def train_perceptron(X, y, perceptron, learning_rate, n_iter):
    errors = np.zeros(n_iter)
    for i in range(n_iter):
        total_error = 0
        for j in range(len(X)):
            x = X[j]
            t = y[j]
            y_pred = perceptron.compute_output(x)
            if y_pred != t:
                error = t - y_pred
                total_error += error ** 2
                perceptron.update_weights(x, error, learning_rate)
        errors[i] = total_error / len(X)
    return errors

# Test du perceptron sur un ensemble de test
def test_perceptron(X_test, y_test, perceptron):
    total_error = 0
    for i in range(len(X_test)):
        x = X_test[i]
        t = y_test[i]
        y_pred = perceptron.compute_output(x)
        if y_pred != t:
            total_error += 1
    return total_error / len(X_test)

# Stockage dans un fichier CSV
def save_points(train_set, test_set):
    with open('points.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(['X1', 'X2', 't'])
        for point in train_set:
            writer.writerow(point)
        for point in test_set:
            writer.writerow(point)

# Generate training data
X_train, y_train = generate_data(100)
X_test, y_test = generate_data(50)

save_points((X_train, y_train), (X_test, y_test))

# Initialize perceptron
perceptron = init_perceptron()

# Train perceptron
learning_rate = 0.1
n_iter = 50
train_errors = train_perceptron(X_train, y_train, perceptron, learning_rate, n_iter)

# Test perceptron
test_error = test_perceptron(X_test, y_test, perceptron)

# Plot training errors
plt.plot(train_errors)
plt.xlabel("Epoch")
plt.ylabel("Training Error")
plt.show()

# Print test error
print("Test error:", test_error)

