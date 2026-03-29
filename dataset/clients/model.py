import numpy as np

class SimpleNN:
    def __init__(self, input_size=1, hidden_size=16, output_size=1, seed=None):
        if seed:
            np.random.seed(seed)
        self.W1  = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1  = np.zeros((1, hidden_size))
        self.W2  = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2  = np.zeros((1, output_size))
        self._a1 = None

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        self._z1 = X @ self.W1 + self.b1
        self._a1 = np.tanh(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = self._sigmoid(self._z2)
        return self._a2

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int).flatten()

    def predict_proba(self, X):
        return self.forward(X).flatten()

    def train(self, X, y, epochs=100, lr=0.01, batch_size=32, verbose=True):
        y = y.reshape(-1, 1)
        n = len(X)
        history = []

        for epoch in range(1, epochs + 1):
            idx      = np.random.permutation(n)
            X_s, y_s = X[idx], y[idx]

            epoch_loss, batches = 0.0, 0

            for start in range(0, n, batch_size):
                Xb  = X_s[start:start + batch_size]
                yb  = y_s[start:start + batch_size]
                m   = len(Xb)

                out = self.forward(Xb)

                loss = -np.mean(yb*np.log(out+1e-8) + (1-yb)*np.log(1-out+1e-8))
                epoch_loss += loss
                batches    += 1

                dL  = -(yb/(out+1e-8)) + ((1-yb)/(1-out+1e-8))
                d2  = dL * out * (1 - out)

                self.W2 -= lr * (self._a1.T @ d2) / m
                self.b2 -= lr * np.mean(d2, axis=0, keepdims=True)

                d1  = (d2 @ self.W2.T) * (1 - self._a1 ** 2)

                self.W1 -= lr * (Xb.T @ d1) / m
                self.b1 -= lr * np.mean(d1, axis=0, keepdims=True)

            avg_loss = epoch_loss / batches
            history.append(avg_loss)

            if verbose and epoch % 20 == 0:
                acc = np.mean(self.predict(X) == y.flatten())
                print(f'  Epoch {epoch:3d}/{epochs} | Loss:{avg_loss:.4f} | Acc:{acc:.4f}')

        return history

    def get_weights(self):
        return {'W1':self.W1.copy(),'b1':self.b1.copy(),
                'W2':self.W2.copy(),'b2':self.b2.copy()}

    def set_weights(self, w):
        self.W1=w['W1'].copy(); self.b1=w['b1'].copy()
        self.W2=w['W2'].copy(); self.b2=w['b2'].copy()