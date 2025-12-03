import numpy as np
from typing import Literal

class Perception():
    def __init__(self, num_features: int):
        self.num_features = num_features
        self.weights = np.zeros((num_features, 1), dtype=np.float16)
        self.bias = np.zeros(1, dtype=np.float16)

    def forward(self, x: np.ndarray):
        linear = np.dot(x, self.weights) + self.bias
        predictions =  np.where(linear > 0., 1, 0)
        return predictions

    def backward(self, x: np.ndarray, y: np.ndarray):
        predictions = self.forward(x)
        errors = y - predictions;
        return errors;

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        for e in range(epochs):
            for i in range(y.shape[0]):
                errors = self.backward(x[i].reshape(1, self.num_features), y[i]).reshape(-1)
                self.weights += (errors * x[i]).reshape(self.num_features, 1)
                self.bias += errors

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> int:
        predictions = self.forward(x).reshape(-1)
        accuracy: int = np.sum(predictions == y) / y.shape[0]
        return accuracy
