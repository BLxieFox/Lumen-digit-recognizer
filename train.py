"""训练脚本: 下载 MNIST 并用 MLP 训练, 保存权重到 model/mnist_mlp.npz。

用法:
    python train.py                          # 默认超参
    python train.py --epochs 15 --batch 128 --lr 0.05

首次运行会自动下载 MNIST 数据集 (缓存到 data/ 目录), 之后完全离线。
"""

import os
import sys
import gzip
import struct
import argparse
import urllib.request

# ---- 允许从项目内 vendor 目录加载依赖 (离线自包含) ----
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import numpy as np

from network import MLP

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "model")

# 文件名 -> 说明
FILES = {
    "train-images-idx3-ubyte.gz": "训练图像",
    "train-labels-idx1-ubyte.gz": "训练标签",
    "t10k-images-idx3-ubyte.gz": "测试图像",
    "t10k-labels-idx1-ubyte.gz": "测试标签",
}

# 多个镜像, 依次尝试
MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


def ensure_mnist(data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    for name in FILES:
        dest = os.path.join(data_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        print(f"[数据] 下载 {name} ...")
        ok = False
        for mirror in MIRRORS:
            url = mirror + name
            tmp = dest + ".part"
            try:
                with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                os.replace(tmp, dest)
                print(f"       完成 ({mirror})")
                ok = True
                break
            except Exception as e:
                print(f"       失败 {mirror}: {e}")
                if os.path.exists(tmp):
                    os.remove(tmp)
        if not ok:
            raise RuntimeError(f"无法下载 {name}, 请检查网络后重试")
    return data_dir


def read_idx(path):
    """解析 IDX 格式 (MNIST 原始格式)。"""
    with gzip.open(path, "rb") as f:
        buf = f.read()
    magic = struct.unpack(">I", buf[:4])[0]
    ndim = magic & 0xFF
    dims = struct.unpack(">" + "I" * ndim, buf[4:4 + 4 * ndim])
    return np.frombuffer(buf, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist(data_dir=DATA_DIR):
    train_x = read_idx(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    train_y = read_idx(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_x = read_idx(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    test_y = read_idx(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    X_tr = train_x.reshape(-1, 784).astype(np.float32) / 255.0
    X_te = test_x.reshape(-1, 784).astype(np.float32) / 255.0
    Y_tr = np.eye(10, dtype=np.float32)[train_y]
    Y_te = np.eye(10, dtype=np.float32)[test_y]
    return X_tr, train_y, Y_tr, X_te, test_y, Y_te


def accuracy(net, X, y, batch=1000):
    correct = 0
    for i in range(0, len(y), batch):
        xb = X[i:i + batch].T
        pred = np.argmax(net.predict_proba(xb), axis=0)
        correct += int(np.sum(pred == y[i:i + batch]))
    return correct / len(y)


def train(args):
    data_dir = ensure_mnist()
    X_tr, y_tr, Y_tr, X_te, y_te, Y_te = load_mnist(data_dir)
    print(f"[数据] 训练集 {X_tr.shape[0]} 张, 测试集 {X_te.shape[0]} 张")

    net = MLP([784, 128, 64, 10], seed=42)
    print(f"[模型] 架构 {net.sizes}, 参数量 {net.num_params:,}")

    rng = np.random.default_rng(0)
    N = X_tr.shape[0]

    for epoch in range(args.epochs):
        perm = rng.permutation(N)
        total_loss = 0.0
        nb = 0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            xb = X_tr[idx].T
            yb = Y_tr[idx].T
            net.train_step(xb, yb, args.lr, args.momentum, args.l2)
            probs = net.predict_proba(xb)
            loss = -float(np.mean(np.sum(yb * np.log(np.clip(probs, 1e-12, 1.0)), axis=0)))
            total_loss += loss
            nb += 1
        acc = accuracy(net, X_te, y_te)
        print(f"[训练] epoch {epoch + 1:>2}/{args.epochs} | loss {total_loss / nb:.4f} | 测试准确率 {acc * 100:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "mnist_mlp.npz")
    net.save(path)
    print(f"[完成] 权重已保存到 {path}")
    return net, acc


def main():
    p = argparse.ArgumentParser(description="训练手写数字识别 MLP")
    p.add_argument("--epochs", type=int, default=10, help="训练轮数")
    p.add_argument("--batch", type=int, default=64, help="批大小")
    p.add_argument("--lr", type=float, default=0.1, help="学习率")
    p.add_argument("--momentum", type=float, default=0.9, help="动量")
    p.add_argument("--l2", type=float, default=0.0, help="L2 正则系数")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
