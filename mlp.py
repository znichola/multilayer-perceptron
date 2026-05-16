import numpy as np
from typing import Protocol, TypeAlias

eps = 1e-8

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


class Layer(Protocol):
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def backwards(self, grad: np.ndarray, lr: float) -> np.ndarray: ...


# Layers

class Dense:
    def __init__(self, in_dem: int, out_dem: int, activation: ActivType) -> None:
        self.W = np.random.randn(in_dem, out_dem) * np.sqrt(2 / in_dem)
        self.b = np.zeros((1, out_dem))
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        z = x @ self.W + self.b
        self.z = z
        return self.activation.forward(z) if self.activation else z

    def backwards(self, grad: np.ndarray, lr: float) -> np.ndarray:
        grad = grad * self.activation.backwards(self.z)
        dW = self.x.T @ grad
        db = grad.sum(axis=0, keepdims=True)
        dx = grad @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx


class Normalize:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std:  np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
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
        return np.ones_like(x) # a no-op, in this implementaiton it should only be on the last layer pared with CCE


# Loss function

class CategoricalCrossentropy:
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def backwards(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return (y_pred - y_true) / y_true.shape[0]


# Evaluation metric

def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    p = y_pred[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)]
    t = y_true[np.arange(len(y_true)), np.argmax(y_true, axis=1)]
    return -(t * np.log(p)).mean()

# Model

class Sequential:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def compile(self, loss: LossType, lr: float = 0.01, epochs: int = 100, batch: int = 10) -> None:
        self.loss_fn = loss
        self.lr = lr
        self.epochs = epochs
        self.batch = batch

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backwards(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backwards(grad, self.lr)

    def fit(self, X: np.ndarray, y: np.ndarray, X_valid: np.ndarray | None = None, y_valid: np.ndarray | None = None) -> dict:
        history: dict[str, list] = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X))
            X_shuffled, y_shuffled = X[indices], y[indices]

            total_loss = 0.0
            correct = 0

            for start in range(0, len(X), self.batch):
                Xb = X_shuffled[start:start + self.batch]
                yb = y_shuffled[start:start + self.batch]

                y_pred_b = self.forward(Xb)
                loss_b = self.loss_fn.forward(yb, y_pred_b)
                grad_b = self.loss_fn.backwards(yb, y_pred_b)
                self.backwards(grad_b)

                total_loss += loss_b * len(Xb)
                correct += (np.argmax(y_pred_b, axis=1) == np.argmax(yb, axis=1)).sum()

            loss = total_loss / len(X)
            acc = correct / len(X)
            history["loss"].append(loss)
            history["acc"].append(acc)

            if X_valid is not None and y_valid is not None:
                val_pred = self.forward(X_valid)
                val_loss = self.loss_fn.forward(y_valid, val_pred)
                val_acc = (np.argmax(val_pred, axis=1) == np.argmax(y_valid, axis=1)).mean()
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"[mlp/sequential/fit] epoch {epoch+1:02d}/{self.epochs} - accuracy: {acc:.3f} loss: {loss:.4f} - val_accuracy: {val_acc:.3f} val_loss: {val_loss:.4f}")

        self.history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
