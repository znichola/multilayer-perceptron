import numpy as np
from typing import Protocol, TypeAlias

import common as cm

# Protocols

class Activation(Protocol):
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def backwards(x: np.ndarray) -> np.ndarray: ...

ActivType: TypeAlias = type[Activation]


class LossFunction(Protocol):
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float: ...
    @staticmethod
    def backwards(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: ...

LossType: TypeAlias = type[LossFunction]


# TODO : should it also be @staticmethod
class Layer(Protocol):
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def backwards(self, grad: np.ndarray, lr: float) -> np.ndarray: ...


# Layers

class Dense:
    def __init__(self, in_dem: int, out_dem: int, activation: ActivType | None = None) -> None:
        self.W = np.random.randn(in_dem, out_dem) * np.sqrt(2 / in_dem)  # He initialization
        self.b = np.zeros((1, out_dem))
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        z = x @ self.W + self.b
        self.z = z
        return self.activation.forward(z) if self.activation else z

    def backwards(self, grad: np.ndarray, lr: float) -> np.ndarray:
        if self.activation:
            grad = grad * self.activation.backwards(self.z)
        dW = self.x.T @ grad
        db = grad.sum(axis=0, keepdims=True)
        dx = grad @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx


class Normalize:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mean = x.mean(axis=0)
        self.std  = x.std(axis=0) + 1e-8
        return (x - self.mean) / self.std

    def backwards(self, grad: np.ndarray, lr: float) -> np.ndarray:
        return grad


# Activations

class ReLU:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def backwards(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Sigmoid:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backwards(x: np.ndarray) -> np.ndarray:
        s = Sigmoid.forward(x) #TODO store and retrieve
        return s * (1 - s)


class SoftMax:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def backwards(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
        s = SoftMax.forward(x)
        return s * (1 - s)

# Loss functions
# Forward is scalar loss value
# Backwards is gradient of the loss w.r.t. y_pred

class CategoricalCrossentropy:
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def backwards(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return (y_pred - y_true) / y_true.shape[0]


class BinaryCrossentropy:
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-8
        return -(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
        ).mean()

    @staticmethod
    def backwards(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / y_true.shape[0]

# Model

class Sequential:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def compile(self, loss: LossType, lr: float = 0.01, epochs: int = 100, batch: int = 10) -> None:
        self.loss_fn = loss
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        # TODO implement batch

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backwards(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backwards(grad, self.lr)

    def fit( self, X: np.ndarray, y: np.ndarray, X_valid: np.ndarray | None = None, y_valid: np.ndarray | None = None,
    ) -> dict:
        history: dict[str, list] = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

        for epoch in range(self.epochs):
            y_pred = self.forward(X)
            loss   = self.loss_fn.forward(y, y_pred)
            grad   = self.loss_fn.backwards(y, y_pred)
            self.backwards(grad)

            acc = (np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)).mean()
            history["loss"].append(loss)
            history["acc"].append(acc)

            if X_valid is not None and y_valid is not None:
                val_pred  = self.forward(X_valid)
                val_loss  = self.loss_fn.forward(y_valid, val_pred)
                val_acc   = (np.argmax(val_pred, axis=1) == np.argmax(y_valid, axis=1)).mean()
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch:>4} | Loss: {loss:.4f} | Acc: {acc:.3f}"
                + (f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
                if X_valid is not None else "")
            )

        self.history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
