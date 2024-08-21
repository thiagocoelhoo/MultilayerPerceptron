from typing import List
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  


def sigmoid_d(x):
    return x * (1 - x)


class Layer:
    def __init__(self, input_size: int, layer_size: int, activation=sigmoid, activation_d=sigmoid_d):
        self.weights = np.random.random((layer_size, input_size))
        self.bias = np.random.random((layer_size, 1))

        self.inp = np.zeros((input_size, 1))
        self.out = np.zeros((layer_size, 1))

        self.activation = activation
        self.activation_d = activation_d        

    def forward(self, x):
        self.inp = np.array(x).reshape((-1, 1))
        self.out = self.activation(self.weights.dot(self.inp) + self.bias)

        return self.out

    def backward(self, error, alpha):
        delta = error * self.activation_d(self.out)
        self.weights += delta.dot(self.inp.T) * alpha
        self.bias += delta * alpha

        return self.weights.T.dot(delta)

class MultilayerPerceptron:
    def __init__(self, hidden_layers: List[Layer]):
        self.layers = hidden_layers

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def adjust(self, error, alpha):
        for layer in reversed(self.layers):
            error = layer.backward(error, alpha)

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def train(self, x_train, y_train, epochs=100, alpha=0.1):
        for i in range(epochs):
            total_error = 0
            for x, y_target in zip(x_train, y_train):
                y_pred = self.predict(x)
                error = y_target - y_pred
                total_error += self.mean_squared_error(y_target, y_pred)
                self.adjust(error, alpha)

            if i % 100 == 0:
                # Calcula o MSE médio para o batch de dados
                average_error = total_error / len(x_train)
                print(f"Iteration {i+1}/{epochs}, MSE: {average_error}")
