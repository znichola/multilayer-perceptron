import numpy as np
from typing import Protocol, TypeAlias

import common as cm


# Type hints

class Activation(Protocol):
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def backwards(x: np.ndarray) -> np.ndarray: ...

ActivType: TypeAlias = type[Activation]

class Layer(Protocol):
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    def backwards(self, grad: np.ndarray, lr: float) -> np.ndarray: ...


# Layers

class Dense:
    def __init__(self, in_dem, out_dem, activation: ActivType | None =None) -> None:
        # self.W = np.random.randn(in_dem, out_dem) * 0.01
        self.W = np.random.randn(in_dem, out_dem) * np.sqrt(2 / in_dem) # He Initializaion
        self.b = np.zeros((1, out_dem))
        self.activation = activation

    def forward(self, x: np.ndarray):
        self.x = x
        z = x @ self.W + self.b
        self.z = z
        return self.activation.forward(z) if self.activation else z

    def backwards(self, grad, lr):
        if self.activation:
            grad = grad * self.activation.backwards(self.z)
        dW = self.x.T @ grad
        db = grad.sum(axis=0, keepdims=True)
        dx = grad @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx


class Normalize:
    def forward(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
        return (x - self.mean) / self.std

    def backwards(self, grad, lr):
        return grad

# Ops

class ReLU:
    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    @staticmethod
    def backwards(x):
        return (x > 0).astype(float)

class Sigmoid:
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backwards(x):
        s = Sigmoid.forward(x)
        return s * (1 - s)

class SoftMax:
    @staticmethod
    def forward(x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def backwards(x):
        # Softmax gradient handled in crossentropy, so return ones
        return np.ones_like(x)


def binary_corssentropy(y_true, y_pred):
    eps = 1e-8
    return -(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)).mean()

def binary_crossentropy_grad(y_true, y_pred):
    return y_pred - y_true


def categorical_crossentropy(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def categorical_crossentropy_grad(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / y_true.shape[0]


# Model

class Sequential:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def compile(self, loss, lr=0.01):
        self.loss = loss
        self.lr = lr
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backwards(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backwards(grad, self.lr)
    
    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss(y, y_pred)
            grad = categorical_crossentropy_grad(y, y_pred)
            self.backwards(grad)

            # if epoch % 1 == 0:
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(y, axis=1)
            acc = (y_pred_labels == y_true_labels).mean()
            print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.3f}")

    def predict(self, X):
        return self.forward(X)


def load_data(path) -> tuple[np.ndarray, np.ndarray]:
    df = cm.load_data(path) 
    if df is None:
        exit(0)

    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

    X = df.drop(columns=["diagnosis"]).to_numpy(dtype=float)
    y = np.zeros((len(df), 2))
    y[:, 0] = (df["diagnosis"] == "M").astype(float)
    y[:, 1] = (df["diagnosis"] == "B").astype(float)

    return X, y


def main():
    X_train, y_train = load_data("train.csv")
    X_test, y_test = load_data("validation.csv")

    print(f"{X_train.shape=}")
    print(f"{y_train.shape=}")

    model = Sequential([
        Normalize(),
        Dense(X_train.shape[1], 24, activation=Sigmoid),
        Dense(24, 24, activation=Sigmoid),
        Dense(24, 24, activation=Sigmoid),
        Dense(24, 2, activation=SoftMax)  # 2 classes
    ])

    model.compile(loss=categorical_crossentropy, lr=0.1)
    model.fit(X_train, y_train, epochs=500)

    preds = model.predict(X_test)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    for p, t in zip(pred_labels[:10], true_labels[:10]):
        print(f"Pred: {p} | True: {t}")


if __name__ == "__main__":
    main()