"""从零实现的多层感知机 (MLP) 神经网络, 仅依赖 NumPy。

包含:
  - He 参数初始化 (适合 ReLU)
  - ReLU 隐藏层激活 + Softmax 输出层
  - 交叉熵损失 + Softmax 的解析梯度 (反向传播)
  - 带动量的小批量随机梯度下降 (SGD)
  - 权重保存 / 加载 (.npz)

这里不依赖 PyTorch / TensorFlow, 完全离线、可解释。
"""

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def softmax(z: np.ndarray) -> np.ndarray:
    """按列做数值稳定的 softmax (z 形状: (类别数, 样本数))."""
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)


class MLP:
    """全连接多层感知机。

    参数
    ----
    sizes : list[int]
        各层神经元数量, 例如 [784, 128, 64, 10]。
    seed : int
        随机种子, 保证训练可复现。
    """

    def __init__(self, sizes, seed=0):
        self.sizes = list(sizes)
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            # He 初始化: 方差 2 / fan_in, 适配 ReLU
            self.weights.append(rng.standard_normal((fan_out, fan_in)) * np.sqrt(2.0 / fan_in))
            self.biases.append(np.zeros((fan_out, 1)))
        # 动量缓存
        self._vel_w = [np.zeros_like(w) for w in self.weights]
        self._vel_b = [np.zeros_like(b) for b in self.biases]

    # ------------------------------------------------------------------ #
    # 前向传播
    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray):
        """前向传播。

        返回 (activations, zs):
          activations 长度 = 层数 + 1 (包含输入)
          zs          长度 = 层数 (激活前的线性组合)
        """
        acts = [x]
        zs = []
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ a + b
            zs.append(z)
            a = softmax(z) if i == len(self.weights) - 1 else relu(z)
            acts.append(a)
        return acts, zs

    def logits(self, x: np.ndarray) -> np.ndarray:
        """返回最后一层未经过 softmax 的 logits (用于温度缩放)."""
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ a + b
            a = z if i == len(self.weights) - 1 else relu(z)
        return a

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0][-1]

    def predict(self, x: np.ndarray):
        """返回预测类别 (x 为多列时返回列表)."""
        return np.argmax(self.predict_proba(x), axis=0).tolist()

    # ------------------------------------------------------------------ #
    # 反向传播 + 训练
    # ------------------------------------------------------------------ #
    def _backprop(self, x: np.ndarray, y: np.ndarray):
        """计算交叉熵损失的梯度。x: (输入维, 批大小), y: (类别数, 批大小) one-hot。"""
        acts, zs = self.forward(x)
        m = x.shape[1]
        delta = acts[-1] - y  # softmax + 交叉熵的简洁梯度
        grad_w = []
        grad_b = []
        for l in range(len(self.weights) - 1, -1, -1):
            grad_w.insert(0, delta @ acts[l].T / m)
            grad_b.insert(0, np.sum(delta, axis=1, keepdims=True) / m)
            if l > 0:
                delta = (self.weights[l].T @ delta) * (zs[l - 1] > 0)  # ReLU 导数
        return grad_w, grad_b

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float, momentum: float = 0.9, l2: float = 0.0):
        grad_w, grad_b = self._backprop(x, y)
        for i in range(len(self.weights)):
            if l2:
                grad_w[i] = grad_w[i] + l2 * self.weights[i]
            self._vel_w[i] = momentum * self._vel_w[i] - lr * grad_w[i]
            self._vel_b[i] = momentum * self._vel_b[i] - lr * grad_b[i]
            self.weights[i] += self._vel_w[i]
            self.biases[i] += self._vel_b[i]

    # ------------------------------------------------------------------ #
    # 保存 / 加载
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        payload = {"sizes": np.array(self.sizes, dtype=np.int64)}
        for i, w in enumerate(self.weights):
            payload[f"w{i}"] = w
        for i, b in enumerate(self.biases):
            payload[f"b{i}"] = b
        np.savez_compressed(path, **payload)

    @staticmethod
    def load(path: str) -> "MLP":
        with np.load(path) as d:
            sizes = d["sizes"].tolist()
            net = MLP(sizes)
            net.weights = [d[f"w{i}"] for i in range(len(sizes) - 1)]
            net.biases = [d[f"b{i}"] for i in range(len(sizes) - 1)]
        net._vel_w = [np.zeros_like(w) for w in net.weights]
        net._vel_b = [np.zeros_like(b) for b in net.biases]
        return net

    @property
    def num_params(self) -> int:
        return int(sum(w.size for w in self.weights) + sum(b.size for b in self.biases))
