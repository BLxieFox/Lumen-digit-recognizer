"""手写数字识别 Web 应用 (Flask 后端)。

功能:
  - 提供 Web UI (static/ 目录)
  - POST /predict  接收画板图像 + 设置, 做 MNIST 风格预处理 + 神经网络推理
  - GET  /logs     通过 SSE 实时推送日志到前端
  - GET  /status   返回模型信息

运行:
    python app.py                 # 默认 http://127.0.0.1:8000
    python app.py --port 5000
"""

import os
import sys
import json
import time
import queue
import argparse
import threading
from collections import deque

# ---- 允许从项目内 vendor 目录加载依赖 (离线自包含) ----
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory

from network import MLP, softmax

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
MODEL_PATH = os.path.join(HERE, "model", "mnist_mlp.npz")

INPUT_SIZE = 112   # 前端采集的图像边长
GRID = 28          # 神经网络输入 28×28
POOL = INPUT_SIZE // GRID  # 下采样块大小


# --------------------------------------------------------------------- #
# 日志中心 (支持 SSE 订阅)
# --------------------------------------------------------------------- #
class EventLog:
    def __init__(self, maxlen=600):
        self._buf = deque(maxlen=maxlen)
        self._subs = set()
        self._lock = threading.Lock()

    def log(self, level, msg):
        evt = {"ts": time.time(), "level": level, "msg": str(msg)}
        with self._lock:
            self._buf.append(evt)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(evt)
            except queue.Full:
                pass

    def subscribe(self):
        q = queue.Queue(maxsize=2000)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs.discard(q)

    def history(self):
        with self._lock:
            return list(self._buf)


logger = EventLog()
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")


# --------------------------------------------------------------------- #
# 模型加载
# --------------------------------------------------------------------- #
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = MLP.load(MODEL_PATH)
        logger.log("ok", "模型已加载: model/mnist_mlp.npz")
        logger.log("info", f"架构 {model.sizes} | 参数量 {model.num_params:,}")
    except Exception as e:  # noqa: BLE001
        logger.log("error", f"模型加载失败: {e}")
        model = None
else:
    logger.log("warn", "未找到模型权重, 请先运行 python train.py 生成 model/mnist_mlp.npz")


# --------------------------------------------------------------------- #
# 预处理 (模拟 MNIST 的标准化流程)
# --------------------------------------------------------------------- #
def _resize_bilinear(img, out_h, out_w):
    h, w = img.shape
    sy = np.linspace(0, h - 1, out_h)
    sx = np.linspace(0, w - 1, out_w)
    y0 = np.floor(sy).astype(np.int32)
    x0 = np.floor(sx).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (sy - y0)[:, None]
    fx = (sx - x0)[None, :]
    return (img[y0][:, x0] * (1 - fy) * (1 - fx)
            + img[y0][:, x1] * (1 - fy) * fx
            + img[y1][:, x0] * fy * (1 - fx)
            + img[y1][:, x1] * fy * fx)


def preprocess(image, settings):
    """将 (112,112) 的墨迹强度数组处理成 (28,28) 的输入张量。"""
    arr = np.asarray(image, dtype=np.float32)
    if arr.shape != (INPUT_SIZE, INPUT_SIZE):
        arr = np.nan_to_num(arr).reshape(INPUT_SIZE, INPUT_SIZE)
    logger.log("info", f"收到图像 {arr.shape[0]}×{arr.shape[1]}")

    # 1) 最大池化下采样到 28×28 (保留细笔画)
    img = arr[:GRID * POOL, :GRID * POOL].reshape(GRID, POOL, GRID, POOL).max(axis=(1, 3))
    logger.log("info", f"最大池化下采样 {INPUT_SIZE}×{INPUT_SIZE} → 28×28")

    # 2) 阈值去噪
    thr = float(settings.get("threshold", 0.0))
    if thr > 0:
        before = int(np.count_nonzero(img > 0))
        img = np.where(img < thr, 0.0, img)
        logger.log("info", f"阈值过滤 thr={thr:.2f} (非零像素 {before} → {int(np.count_nonzero(img > 0))})")

    # 3) 中心化 (按包围盒裁剪)
    if settings.get("center", True):
        ys, xs = np.nonzero(img > 0)
        if len(xs) == 0:
            logger.log("warn", "画板为空, 无法识别")
            return img, False
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        logger.log("info", f"包围盒 x[{x0}..{x1}] y[{y0}..{y1}] 尺寸 {x1 - x0 + 1}×{y1 - y0 + 1}")
        img = img[y0:y1 + 1, x0:x1 + 1]

    # 4) 尺寸归一化到 20×20 并居中
    if settings.get("normalize", True):
        img = _resize_bilinear(img, 20, 20)
        canvas = np.zeros((GRID, GRID), dtype=np.float32)
        canvas[4:24, 4:24] = img
        img = canvas
        logger.log("info", "尺寸归一化到 20×20 并居中 (MNIST 标准)")
    elif img.shape != (GRID, GRID):
        img = _resize_bilinear(img, GRID, GRID)

    # 5) 数值归一化
    mx = float(img.max())
    if mx > 1.0:
        img = img / mx
    logger.log("info", f"输入就绪: 形状 {img.shape}, 非零像素 {int(np.count_nonzero(img > 0))}")

    return img, True


# --------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/status")
def status():
    return jsonify({
        "model_loaded": model is not None,
        "sizes": model.sizes if model else None,
        "num_params": model.num_params if model else None,
        "input_size": INPUT_SIZE,
        "grid": GRID,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        logger.log("error", "模型未加载, 无法预测")
        return jsonify({"ok": False, "error": "模型未加载, 请先运行 python train.py"}), 503

    t0 = time.perf_counter()
    data = request.get_json(force=True, silent=True) or {}
    image = data.get("image")
    settings = data.get("settings", {})

    if image is None:
        return jsonify({"ok": False, "error": "缺少 image 字段"}), 400

    try:
        x, ok = preprocess(image, settings)
    except Exception as e:  # noqa: BLE001
        logger.log("error", f"预处理失败: {e}")
        return jsonify({"ok": False, "error": f"预处理失败: {e}"}), 500

    if not ok:
        return jsonify({"ok": False, "empty": True, "error": "画板为空"})

    # 前向传播
    x = x.reshape(-1, 1)
    z = model.logits(x)

    temperature = float(settings.get("temperature", 1.0))
    temperature = max(0.1, min(temperature, 10.0))
    probs = softmax(z / temperature).ravel()
    logger.log("info", f"前向传播完成 (temperature={temperature:.2f})")
    logger.log("info", f"logits 范围 [{float(z.min()):.2f}, {float(z.max()):.2f}]")

    order = np.argsort(probs)[::-1]
    top_k = int(settings.get("top_k", 10))
    top_k = max(1, min(top_k, 10))
    probabilities = [
        {"digit": int(d), "prob": round(float(probs[d]), 4)}
        for d in order[:top_k]
    ]
    digit = int(order[0])
    confidence = float(probs[order[0]])

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.log("ok", f"识别结果: {digit} (置信度 {confidence * 100:.1f}%, 用时 {elapsed_ms:.1f} ms)")

    return jsonify({
        "ok": True,
        "digit": digit,
        "confidence": confidence,
        "probabilities": probabilities,
        "elapsed_ms": round(elapsed_ms, 2),
    })


@app.route("/logs")
def logs():
    def gen():
        q = logger.subscribe()
        try:
            for evt in logger.history():
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
            while True:
                try:
                    evt = q.get(timeout=15)
                    yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            logger.unsubscribe(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main():
    p = argparse.ArgumentParser(description="手写数字识别 Web 应用")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    logger.log("info", f"启动服务 http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
