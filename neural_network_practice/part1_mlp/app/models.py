from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np

from exceptions import InvalidLayerSizeError, MismatchedDataError


class NeuralNetwork:
    """Fully connected feed-forward network with configurable activation."""

    SUPPORTED_ACTIVATIONS = {"sigmoid", "relu"}

    def __init__(self, layer_sizes: Sequence[int], activation: str = "relu", seed: int | None = None):
        if len(layer_sizes) < 3 or any(int(size) <= 0 for size in layer_sizes):
            raise InvalidLayerSizeError("Нужно минимум три слоя: входной, скрытый и выходной.")
        activation = activation.lower().strip()
        if activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Неподдерживаемая активация: {activation}")
        self.layer_sizes = [int(size) for size in layer_sizes]
        self.activation_name = activation
        self.rng = np.random.default_rng(seed)
        self.weights = [self.rng.normal(0, np.sqrt(2 / self.layer_sizes[i]), (self.layer_sizes[i], self.layer_sizes[i + 1]))
                        for i in range(len(self.layer_sizes) - 1)]
        self.biases = [np.zeros((1, size)) for size in self.layer_sizes[1:]]
        self.loss_history: List[float] = []

    def _activation(self, x):
        if self.activation_name == "sigmoid":
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        return np.maximum(0, x)

    def _activation_derivative(self, x):
        if self.activation_name == "sigmoid":
            s = self._activation(x)
            return s * (1 - s)
        return (x > 0).astype(float)

    def forward(self, X, return_cache=False):
        a = np.asarray(X, dtype=float)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.ndim != 2 or a.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError("Размерность входных признаков не совпадает с размером входного слоя.")
        activations = [a]
        pre_activations = []
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ weight + bias
            pre_activations.append(z)
            activations.append(self._activation(z) if index < len(self.weights) - 1 else z)
        return (activations[-1], activations, pre_activations) if return_cache else activations[-1]

    def predict(self, X):
        return self.forward(X)

    def train(self, X, y, epochs=100, learning_rate=0.01, batch_size=32, shuffle=True):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim not in (1, 2) or len(X) != len(y):
            raise MismatchedDataError("X и y должны иметь одинаковое число примеров.")
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        if y.shape[1] != self.layer_sizes[-1]:
            raise MismatchedDataError("Размерность целевой переменной не совпадает с выходным слоем.")
        if epochs <= 0 or learning_rate <= 0 or batch_size <= 0:
            raise ValueError("epochs, learning_rate и batch_size должны быть положительными.")
        for _ in range(epochs):
            indices = self.rng.permutation(len(X)) if shuffle else np.arange(len(X))
            for start in range(0, len(X), batch_size):
                batch = indices[start:start + batch_size]
                output, activations, zs = self.forward(X[batch], return_cache=True)
                delta = 2 * (output - y[batch]) / len(batch)
                grad_w, grad_b = [None] * len(self.weights), [None] * len(self.biases)
                for layer in range(len(self.weights) - 1, -1, -1):
                    grad_w[layer] = activations[layer].T @ delta
                    grad_b[layer] = delta.sum(axis=0, keepdims=True)
                    if layer:
                        delta = (delta @ self.weights[layer].T) * self._activation_derivative(zs[layer - 1])
                for layer in range(len(self.weights)):
                    self.weights[layer] -= learning_rate * grad_w[layer]
                    self.biases[layer] -= learning_rate * grad_b[layer]
            self.loss_history.append(float(np.mean((self.forward(X) - y) ** 2)))
        return self.loss_history

    def save(self, path):
        payload = {"layer_sizes": self.layer_sizes, "activation": self.activation_name,
                   "weights": [w.tolist() for w in self.weights], "biases": [b.tolist() for b in self.biases],
                   "loss_history": self.loss_history}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(payload["layer_sizes"], payload["activation"])
        model.weights = [np.asarray(w, dtype=float) for w in payload["weights"]]
        model.biases = [np.asarray(b, dtype=float) for b in payload["biases"]]
        model.loss_history = list(payload.get("loss_history", []))
        return model