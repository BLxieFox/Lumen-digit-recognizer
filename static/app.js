/* 手写数字识别 · 前端逻辑 */

(() => {
  "use strict";

  const INPUT_SIZE = 112;   // 发送给后端的图像边长
  const BOARD_PX = 560;     // 画板底层像素 (INPUT_SIZE 的整数倍)
  const SCALE = BOARD_PX / INPUT_SIZE;

  const board = document.getElementById("board");
  const hint = document.getElementById("board-hint");
  const bigDigit = document.getElementById("big-digit");
  const confidenceEl = document.getElementById("confidence");
  const barsEl = document.getElementById("bars");
  const metaEl = document.getElementById("meta");
  const logEl = document.getElementById("log");

  // ---------- 画板 ----------
  board.width = BOARD_PX;
  board.height = BOARD_PX;
  const ctx = board.getContext("2d");

  const INK = "#1b1b1b";
  let strokes = [];          // { width, points: [[x,y], ...] }
  let current = null;        // 当前正在画的笔画
  let drawing = false;

  function fillPaper() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, BOARD_PX, BOARD_PX);
  }

  function brushSize() {
    return parseInt(document.getElementById("brush").value, 10);
  }

  function redrawAll() {
    fillPaper();
    ctx.strokeStyle = INK;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const s of strokes) {
      ctx.lineWidth = s.width;
      ctx.beginPath();
      ctx.moveTo(s.points[0][0], s.points[0][1]);
      for (let i = 1; i < s.points.length; i++) {
        ctx.lineTo(s.points[i][0], s.points[i][1]);
      }
      ctx.stroke();
    }
    hint.classList.toggle("hidden", strokes.length > 0);
  }

  function pos(e) {
    const r = board.getBoundingClientRect();
    return [
      (e.clientX - r.left) * (BOARD_PX / r.width),
      (e.clientY - r.top) * (BOARD_PX / r.height),
    ];
  }

  board.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    board.setPointerCapture(e.pointerId);
    drawing = true;
    current = { width: brushSize(), points: [pos(e)] };
    const [x, y] = current.points[0];
    ctx.strokeStyle = INK;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = current.width;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 0.01, y + 0.01); // 点一下也要留一个点
    ctx.stroke();
    hint.classList.add("hidden");
  });

  board.addEventListener("pointermove", (e) => {
    if (!drawing || !current) return;
    const [x, y] = pos(e);
    current.points.push([x, y]);
    ctx.lineWidth = current.width;
    ctx.strokeStyle = INK;
    ctx.lineTo(x, y);
    ctx.stroke();
  });

  function finishStroke() {
    if (!drawing) return;
    drawing = false;
    if (current && current.points.length) {
      strokes.push(current);
    }
    current = null;
    if (document.getElementById("set-auto").checked && strokes.length) {
      schedulePredict();
    }
  }

  board.addEventListener("pointerup", finishStroke);
  board.addEventListener("pointercancel", finishStroke);

  // ---------- 采样: 画板 → 112×112 墨迹强度 ----------
  function captureImage() {
    const data = ctx.getImageData(0, 0, BOARD_PX, BOARD_PX).data;
    const ink = new Array(INPUT_SIZE);
    for (let y = 0; y < INPUT_SIZE; y++) {
      ink[y] = new Array(INPUT_SIZE);
      for (let x = 0; x < INPUT_SIZE; x++) {
        let mx = 0;
        for (let by = 0; by < SCALE; by++) {
          for (let bx = 0; bx < SCALE; bx++) {
            const px = (y * SCALE + by) * BOARD_PX + (x * SCALE + bx);
            const i = px * 4;
            const g = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            const v = 1 - g / 255; // 墨迹=1, 纸张=0
            if (v > mx) mx = v;
          }
        }
        ink[y][x] = +mx.toFixed(4);
      }
    }
    return ink;
  }

  // ---------- 设置读取 ----------
  function getSettings() {
    return {
      center: document.getElementById("set-center").checked,
      normalize: document.getElementById("set-normalize").checked,
      threshold: parseFloat(document.getElementById("set-threshold").value),
      temperature: parseFloat(document.getElementById("set-temperature").value),
      top_k: parseInt(document.getElementById("set-topk").value, 10),
      confidence_threshold: parseFloat(document.getElementById("set-conf").value),
    };
  }

  // ---------- 预测 ----------
  let predicting = false;
  let debounceTimer = null;

  function schedulePredict() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(predict, 220);
  }

  async function predict() {
    if (predicting) return;
    if (!strokes.length) {
      metaEl.textContent = "画板为空，请先书写";
      return;
    }
    predicting = true;
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: captureImage(), settings: getSettings() }),
      });
      const data = await res.json();
      if (data.ok) {
        renderResult(data);
      } else {
        bigDigit.textContent = "?";
        confidenceEl.textContent = "—";
        barsEl.innerHTML = `<div class="bar-placeholder">${data.error || "无法识别"}</div>`;
        metaEl.textContent = data.error || "";
      }
    } catch (err) {
      metaEl.textContent = "请求失败: " + err.message;
    } finally {
      predicting = false;
    }
  }

  function renderResult(data) {
    const conf = data.confidence;
    bigDigit.textContent = String(data.digit);
    bigDigit.classList.remove("pop");
    void bigDigit.offsetWidth; // 重新触发动画
    bigDigit.classList.add("pop");
    bigDigit.classList.toggle("uncertain", conf < getSettings().confidence_threshold);

    confidenceEl.textContent = (conf * 100).toFixed(1) + "%";

    const bars = data.probabilities
      .map((p, i) => `
        <div class="bar-row ${i === 0 ? "top" : ""}">
          <span class="bar-label">${p.digit}</span>
          <span class="bar-track"><span class="bar-fill" style="width:${(p.prob * 100).toFixed(1)}%"></span></span>
          <span class="bar-pct">${(p.prob * 100).toFixed(1)}%</span>
        </div>`)
      .join("");
    barsEl.innerHTML = bars;

    metaEl.textContent = `前向传播用时 ${data.elapsed_ms.toFixed(2)} ms · 温度 T=${getSettings().temperature.toFixed(1)}`;
  }

  // ---------- 日志 (SSE) ----------
  function connectLogs() {
    const es = new EventSource("/logs");
    es.onopen = () => setStatus(true, "已连接");
    es.onerror = () => setStatus(false, "连接断开，重连中…");
    es.onmessage = (e) => {
      try {
        const evt = JSON.parse(e.data);
        appendLog(evt);
      } catch (_) { /* ignore */ }
    };
  }

  function appendLog(evt) {
    const t = new Date(evt.ts * 1000);
    const hh = String(t.getHours()).padStart(2, "0");
    const mm = String(t.getMinutes()).padStart(2, "0");
    const ss = String(t.getSeconds()).padStart(2, "0");
    const time = `${hh}:${mm}:${ss}`;

    const row = document.createElement("div");
    row.className = "log-entry";
    row.innerHTML = `
      <span class="log-time">${time}</span>
      <span class="log-level ${evt.level}">${evt.level}</span>
      <span class="log-msg"></span>`;

    const msg = document.createElement("span");
    msg.className = "log-msg";
    msg.textContent = evt.msg;
    row.querySelector(".log-msg").replaceWith(msg);

    const nearBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 60;
    logEl.appendChild(row);
    if (nearBottom) logEl.scrollTop = logEl.scrollHeight;
  }

  function setStatus(on, text) {
    const dot = document.getElementById("status-dot");
    dot.classList.toggle("on", on);
    dot.classList.toggle("off", !on);
    document.getElementById("status-text").textContent = text;
  }

  // ---------- 模型状态 ----------
  async function loadStatus() {
    try {
      const res = await fetch("/status");
      const s = await res.json();
      const chip = document.getElementById("model-chip");
      if (s.model_loaded) {
        chip.textContent = `MLP ${s.sizes.join("-")} · ${s.num_params.toLocaleString()} 参数`;
      } else {
        chip.textContent = "未加载模型";
      }
    } catch (_) { /* ignore */ }
  }

  // ---------- 控件绑定 ----------
  document.getElementById("btn-clear").addEventListener("click", () => {
    strokes = [];
    current = null;
    redrawAll();
    bigDigit.textContent = "?";
    confidenceEl.textContent = "—";
    barsEl.innerHTML = `<div class="bar-placeholder">书写后自动显示各数字概率</div>`;
    metaEl.textContent = "等待输入…";
  });

  document.getElementById("btn-undo").addEventListener("click", () => {
    strokes.pop();
    current = null;
    redrawAll();
    if (document.getElementById("set-auto").checked && strokes.length) schedulePredict();
  });

  document.getElementById("btn-clear-log").addEventListener("click", () => {
    logEl.innerHTML = "";
  });

  const brushInput = document.getElementById("brush");
  brushInput.addEventListener("input", () => {
    document.getElementById("brush-val").textContent = brushInput.value;
  });

  // 滑块数值回显
  [["set-threshold", "thr-val", 2], ["set-temperature", "temp-val", 1],
   ["set-topk", "topk-val", 0], ["set-conf", "conf-val", 2]].forEach(([id, val, dec]) => {
    const el = document.getElementById(id);
    const show = () => { document.getElementById(val).textContent = parseFloat(el.value).toFixed(dec); };
    el.addEventListener("input", show);
    show();
  });

  // 设置变化时若自动识别则重预测
  ["set-threshold", "set-temperature", "set-center", "set-normalize"].forEach((id) => {
    document.getElementById(id).addEventListener("input", () => {
      if (document.getElementById("set-auto").checked && strokes.length) schedulePredict();
    });
  });

  // ---------- 初始化 ----------
  fillPaper();
  redrawAll();
  connectLogs();
  loadStatus();
})();
