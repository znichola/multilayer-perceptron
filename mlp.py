import numpy as np
from typing import Protocol, TypeAlias

eps = 1e-8


class Activation(Protocol):
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def backwards(x: np.ndarray) -> np.ndarray: ...


ActivType: TypeAlias = type[Activation]


class Initialiser(Protocol):
    @staticmethod
    def init(in_dim: int, out_dim: int) -> np.ndarray: ...


InitType: TypeAlias = type[Initialiser]


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
    def __init__(self, in_dem: int, out_dem: int, activation: ActivType, weight_init: InitType) -> None:
        self.W = weight_init.init(in_dem, out_dem)
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
        s = Sigmoid.forward(x) # TODO store and retrieve
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


# Weight initialiser

class HeUniform:
    @staticmethod
    def init(in_dim: int, out_dim: int) -> np.ndarray:
        limit = np.sqrt(6 / in_dim)
        return np.random.uniform(-limit, limit, (in_dim, out_dim))


class LeCunUniform:
    @staticmethod
    def init(in_dim: int, out_dim: int) -> np.ndarray:
        limit = np.sqrt(3 / in_dim)
        return np.random.uniform(-limit, limit, (in_dim, out_dim))


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

def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.floating:
    # E = -1/N * sum(y*log(p) + (1-y)*log(1-p)) with p = P(malignant)
    p = np.clip(y_pred[:, 1], eps, 1 - eps)
    y = y_true[:, 1]
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# Early stopping

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0
        self.best_weights: list[tuple[np.ndarray, np.ndarray]] | None = None

    def step(self, val_loss: float, epoch: int, layers: list) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            self.best_weights = [(l.W.copy(), l.b.copy()) for l in layers]
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"[mlp/fit] early stopping at epoch {epoch+1}, restoring weights from best epoch (val_loss={self.best_loss:.4f})")
                return True
        return False

    def restore(self, layers: list) -> None:
        if self.best_weights is None:
            return
        for l, (W, b) in zip(layers, self.best_weights):
            l.W, l.b = W, b


# Model

class Sequential:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers
        self.mean: None | np.ndarray = None
        self.std: None | np.ndarray = None

    def normalize(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None and self.std is None:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0) + 1e-8
        return (X - self.mean) / self.std

    def compile(self, loss: LossType, lr: float = 0.01, epochs: int = 100, batch: int = 50, patience: int = 0, min_delta: float = 1e-4) -> None:
        self.loss_fn = loss
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.patience = patience
        self.min_delta = min_delta

    def __str__(self) -> str:
        return f"Sequential({len(self.layers)} layers) epochs={self.epochs} batch={self.batch} lr={self.lr} patience={self.patience} min_delta={self.min_delta}"

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backwards(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backwards(grad, self.lr)

    def fit_epoch(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        total_loss, correct = 0.0, 0
        self.batch = len(X) if self.batch == 0 else self.batch
        for start in range(0, len(X), self.batch):
            Xb, yb = X[start:start + self.batch], y[start:start + self.batch]
            y_pred = self.forward(Xb)
            total_loss += self.loss_fn.forward(yb, y_pred) * len(Xb)
            correct += (np.argmax(y_pred, axis=1) == np.argmax(yb, axis=1)).sum()
            self.backwards(self.loss_fn.backwards(yb, y_pred))
        return total_loss / len(X), correct / len(X)

    def fit(self, X: np.ndarray, y: np.ndarray, X_valid: np.ndarray | None = None, y_valid: np.ndarray | None = None) -> dict:
        history: dict[str, list] = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
        es = None
        if self.patience > 0 and X_valid is not None:
            es = EarlyStopping(self.patience, self.min_delta)

        X = self.normalize(X)
        if X_valid is not None:
            X_valid = self.normalize(X_valid)

        for epoch in range(self.epochs):
            loss, acc = self.fit_epoch(X, y)

            if X_valid is not None and y_valid is not None:
                val_pred = self.forward(X_valid)
                val_loss = self.loss_fn.forward(y_valid, val_pred)
                val_acc  = (np.argmax(val_pred, axis=1) == np.argmax(y_valid, axis=1)).mean()
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history["loss"].append(loss)
            history["acc"].append(acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"[mlp/fit] epoch {epoch+1:02d}/{self.epochs} - accuracy: {acc:.3f} loss: {loss:.4f} - val_accuracy: {val_acc:.3f} val_loss: {val_loss:.4f}")

            if es and es.step(val_loss, epoch, self.layers):
                es.restore(self.layers)
                break
        else:
            if es:
                es.restore(self.layers)

        self.history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self.normalize(X)
        return self.forward(X)
