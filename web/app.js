import init, { preprocess } from "../pkg/ocr.js";

const SIDE = 28;
const LEN = SIDE * SIDE;
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// Global App State
const state = {
  activeTab: "playground",
  datasetType: "digits", // 'digits' | 'letters' | 'alphanumeric'
  trainSize: 10000,
  fullTest: false,
  numClasses: 10,
  testImages: null,
  testLabels: null,
  testIndex: 0,
  currentImage: new Uint8Array(LEN),
  modelData: null,
  accuracy: {},
  trainMs: {},
  confusion: null,
  disagreements: [],
  requestId: 0,
  baselineData: null,
};

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

function classLabel(idx) {
  if (state.datasetType === "digits") return String(idx);
  if (state.datasetType === "letters") return String.fromCharCode(65 + idx);
  // Alphanumeric: 0..9 are digits, 10..35 are A..Z
  if (idx < 10) return String(idx);
  return String.fromCharCode(65 + (idx - 10));
}

// Code files content cache
const rustFiles = {
  centroid: {
    title: "src/centroid.rs — Class Mean Distance",
    rationale: "Zero hyperparameter model. Dynamically infers num_classes from dataset labels. Computes class prototypes in f64 to avoid overflow.",
    code: `pub fn train(images: &[Image], labels: &[u8]) -> CentroidModel {
    let num_classes = infer_num_classes(labels);
    let counts = class_counts(labels, num_classes);
    let mut sums = vec![0.0f64; num_classes * IMAGE_LEN];
    for (image, &label) in images.iter().zip(labels) {
        let row = &mut sums[label as usize * IMAGE_LEN..(label as usize + 1) * IMAGE_LEN];
        for (s, &p) in row.iter_mut().zip(image.pixels()) {
            *s += p as f64;
        }
    }
    CentroidModel { num_classes, means, counts }
}`
  },
  knn: {
    title: "src/knn.rs — Geometric Nearest Neighbors",
    rationale: "Non-parametric. Maintains sorted list of top K=3 neighbors by Euclidean distance. Majority label wins.",
    code: `pub fn predict(model: &KnnModel, image: &Image) -> KnnEvidence {
    let mut nearest: Vec<(u32, usize)> = Vec::with_capacity(K + 1);
    for (index, train) in model.images.chunks_exact(IMAGE_LEN).enumerate() {
        let d = squared_distance(image.pixels(), train);
        if nearest.len() == K && d >= nearest[K - 1].0 {
            continue;
        }
        let pos = nearest.iter().position(|&(nd, _)| d < nd).unwrap_or(nearest.len());
        nearest.insert(pos, (d, index));
        nearest.truncate(K);
    }
}`
  },
  perceptron: {
    title: "src/perceptron.rs — Linear Softmax",
    rationale: "C affine hyperplanes. Cross-entropy log loss with hand-written gradient. Numerically stable softmax via max subtraction.",
    code: `pub fn train(images: &[Image], labels: &[u8], seed: u64) -> PerceptronModel {
    let num_classes = infer_num_classes(labels);
    for _ in 0..EPOCHS {
        for batch in order.chunks(BATCH_SIZE) {
            for &n in batch {
                let xn = &x[n * IMAGE_LEN..(n + 1) * IMAGE_LEN];
                let p = softmax(&logits(&model, xn));
                for c in 0..num_classes {
                    let dz = (p[c] - if c == labels[n] as usize { 1.0 } else { 0.0 }) * inv_n;
                    grad_b[c] += dz;
                    for (g, &xi) in grad_w[c * IMAGE_LEN..(c + 1) * IMAGE_LEN].iter_mut().zip(xn) {
                        *g += dz * xi;
                    }
                }
            }
        }
    }
}`
  },
  mlp: {
    title: "src/mlp.rs — Hand-Written Backpropagation with ReLU",
    rationale: "784→64→C architecture. He initialization (±sqrt(6/fan_in)). ReLU derivative is step function [z > 0], preventing gradient vanishing.",
    code: `// Backpropagation pass inside batch loop:
let (z1, a, z2) = forward(model, xn);
let p = softmax(&z2);
for c in 0..model.num_classes {
    dz2[c] = (p[c] - if c == labels[n] as usize { 1.0 } else { 0.0 }) * inv_n;
    for h in 0..HIDDEN {
        g.w2[c * HIDDEN + h] += dz2[c] * a[h];
    }
}
for h in 0..HIDDEN {
    if z1[h] <= 0.0 { continue; } // ReLU gradient gate
    let mut dz1 = 0.0;
    for c in 0..model.num_classes {
        dz1 += dz2[c] * model.w2[c * HIDDEN + h];
    }
    for (gw, &xi) in g.w1[h * IMAGE_LEN..(h + 1) * IMAGE_LEN].iter_mut().zip(xn) {
        *gw += dz1 * xi;
    }
}`
  },
  preprocess: {
    title: "src/preprocess.rs — Integer-Exact Preprocessing",
    rationale: "Replicates MNIST pipeline with zero float rounding drift: integer grayscale, bounding box crop, 20x20 box filter, center of mass centering.",
    code: `// Center of mass calculation
for y in 0..BOX_SIDE {
    for x in 0..BOX_SIDE {
        let v = small[y * BOX_SIDE + x] as u64;
        mass += v;
        mx += v * x as u64;
        my += v * y as u64;
    }
}
let (cx, cy) = (((mx + mass / 2) / mass), ((my + mass / 2) / mass));
let (dx, dy) = (14 - cx, 14 - cy); // Center on coordinate (14, 14)`
  },
  onnx: {
    title: "src/onnx.rs — Custom Protobuf Wire Serializer",
    rationale: "Encodes ModelProto and GraphProto directly into bytes without prost/protobuf crates. Native compliance with Netron & ONNX Runtime.",
    code: `pub fn write_varint(&mut self, mut val: u64) {
    while val >= 0x80 {
        self.buf.push(((val & 0x7F) as u8) | 0x80);
        val >>= 7;
    }
    self.buf.push((val & 0x7F) as u8);
}`
  }
};

// Canvas & Drawing
function drawGray(canvas, pixels, side = SIDE) {
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(side, side);
  for (let i = 0; i < side * side; i++) {
    const v = pixels[i];
    img.data.set([v, v, v, 255], i * 4);
  }
  const off = new OffscreenCanvas(side, side);
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

function drawSigned(canvas, values) {
  const maxAbs = values.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1;
  const px = new Uint8ClampedArray(LEN * 4);
  for (let i = 0; i < LEN; i++) {
    const t = Math.max(-1, Math.min(1, values[i] / maxAbs));
    const to = t >= 0 ? [217, 119, 6] : [37, 99, 235];
    const a = Math.abs(t);
    const r = Math.round(245 + (to[0] - 245) * a);
    const g = Math.round(245 + (to[1] - 245) * a);
    const b = Math.round(245 + (to[2] - 245) * a);
    px.set([r, g, b, 255], i * 4);
  }
  const off = new OffscreenCanvas(SIDE, SIDE);
  off.getContext("2d").putImageData(new ImageData(px, SIDE, SIDE), 0, 0);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

function setupDrawing() {
  const canvas = $("#draw-canvas");
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  let drawing = false;
  let last = null;

  const getPos = (e) => {
    const rect = canvas.getBoundingClientRect();
    return [
      ((e.clientX - rect.left) * canvas.width) / rect.width,
      ((e.clientY - rect.top) * canvas.height) / rect.height
    ];
  };

  const drawStroke = (p) => {
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(last[0], last[1]);
    ctx.lineTo(p[0], p[1]);
    ctx.stroke();
    last = p;
  };

  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    drawing = true;
    last = getPos(e);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const p = getPos(e);
    drawStroke(p);
    processDrawnImage();
  });

  const stop = () => { drawing = false; last = null; };
  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointercancel", stop);

  $("#clear-draw-btn").onclick = () => {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    processDrawnImage();
  };
}

function processDrawnImage() {
  const canvas = $("#draw-canvas");
  const ctx = canvas.getContext("2d");
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const rawBytes = new Uint8Array(imgData.data.buffer);
  const processed = preprocess(rawBytes, canvas.width, canvas.height);
  $("#test-sample-meta").textContent = "Drawing on canvas (Custom Input)";
  setInputImage(processed);
}

function setInputImage(pixels) {
  state.currentImage = pixels;
  drawGray($("#input-grid-canvas"), pixels);
  if (state.modelData) {
    const id = ++state.requestId;
    worker.postMessage({ type: "predict", id, image: pixels });
  }
}

// Router & Tabs
function setupRouter() {
  const switchTab = (tab) => {
    state.activeTab = tab;
    $$(".nav-link").forEach((a) => a.classList.toggle("active", a.dataset.tab === tab));
    $$(".tab-view").forEach((el) => el.classList.toggle("active", el.id === `tab-${tab}`));
    if (tab === "experiments") renderCharts();
  };

  window.addEventListener("hashchange", () => {
    const route = location.hash.replace("#/", "").split("/")[0] || "playground";
    switchTab(route);
  });

  $$(".nav-link").forEach((a) => {
    a.onclick = (e) => {
      e.preventDefault();
      location.hash = `#/${a.dataset.tab}`;
    };
  });

  const initial = location.hash.replace("#/", "").split("/")[0] || "playground";
  switchTab(initial);
}

// Training Setup Modal & Controls
function setupTrainingModal() {
  const modal = $("#modal-training-setup");
  $("#open-train-modal-btn").onclick = () => modal.showModal();

  // Dataset radio changes
  $$("input[name='dataset-choice']").forEach((radio) => {
    radio.onchange = () => {
      const choice = radio.value;
      const size60k = $("#size-60k-btn");
      if (choice === "digits") {
        size60k.style.display = "inline-block";
      } else {
        size60k.style.display = "none";
        if (state.trainSize === 60000) {
          state.trainSize = 10000;
          $$(".modal-size-btn").forEach(b => b.classList.toggle("active", b.dataset.size === "10000"));
        }
      }
      updateWarningText();
    };
  });

  // Size buttons in modal
  $$(".modal-size-btn").forEach((btn) => {
    btn.onclick = () => {
      $$(".modal-size-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.trainSize = Number(btn.dataset.size);
      updateWarningText();
    };
  });

  function updateWarningText() {
    const choice = $('input[name="dataset-choice"]:checked').value;
    const sz = state.trainSize;
    let est = "~4 seconds";
    if (sz === 10) est = "< 0.1s";
    else if (sz === 100) est = "~0.2s";
    else if (sz === 1000) est = "~0.5s";
    else if (sz === 60000) est = "~25 seconds (may warm CPU)";

    $("#time-warning-text").textContent = `⚡ Training on ${sz} samples (${choice}): estimated time ${est}.`;
  }

  $("#start-train-btn").onclick = async () => {
    const choice = $('input[name="dataset-choice"]:checked').value;
    const fullTest = $("#modal-full-test-check").checked;
    state.datasetType = choice;
    state.fullTest = fullTest;

    $("#train-progress-block").style.display = "block";
    $("#start-train-btn").disabled = true;

    worker.postMessage({
      type: "retrain",
      datasetType: state.datasetType,
      size: state.trainSize,
      fullTest: state.fullTest,
    });
  };
}

// Worker message handler
worker.onmessage = async ({ data }) => {
  const pill = $("#status-pill");
  if (data.type === "progress") {
    pill.className = "status-pill";
    pill.textContent = data.text;
    $("#train-progress-status").textContent = data.text;
    $("#train-progress-pct").textContent = `${data.pct}%`;
    $("#train-progress-fill").style.width = `${data.pct}%`;
  } else if (data.type === "error") {
    pill.className = "status-pill error";
    pill.textContent = data.text;
    $("#train-progress-block").style.display = "none";
    $("#start-train-btn").disabled = false;
  } else if (data.type === "ready") {
    pill.className = "status-pill ready";
    pill.textContent = `Trained on ${data.trainSize} samples · Ready`;
    state.numClasses = data.numClasses;
    state.modelData = data.modelData;
    state.accuracy = data.accuracy;
    state.trainMs = data.trainMs;
    state.confusion = data.confusion;
    state.disagreements = data.disagreements;

    $("#active-profile-chip").textContent = `Profile: ${state.datasetType.toUpperCase()} · ${data.trainSize} samples`;
    $("#train-progress-block").style.display = "none";
    $("#start-train-btn").disabled = false;
    $("#modal-training-setup").close();

    // Reload test pool for the active dataset
    await loadTestPool();
    updateAccuracyChips();
    setInputImage(state.currentImage);
    if (state.activeTab === "experiments") renderCharts();
  } else if (data.type === "prediction" && data.id === state.requestId) {
    renderPredictionResults(data.results);
  } else if (data.type === "onnx_exported") {
    downloadBlob(data.bytes, `${data.method}_${state.datasetType}.onnx`);
  }
};

async function loadTestPool() {
  let imgPath, lblPath;
  if (state.datasetType === "digits") {
    imgPath = "data/test-1000.u8";
    lblPath = "data/test-1000-labels.u8";
  } else if (state.datasetType === "letters") {
    imgPath = "data/test-letters-1000.u8";
    lblPath = "data/test-letters-1000-labels.u8";
  } else {
    imgPath = "data/test-alphanumeric-1000.u8";
    lblPath = "data/test-alphanumeric-1000-labels.u8";
  }

  const [tImgs, tLbls] = await Promise.all([
    fetch(imgPath).then(r => r.arrayBuffer()),
    fetch(lblPath).then(r => r.arrayBuffer()),
  ]);
  state.testImages = new Uint8Array(tImgs);
  state.testLabels = new Uint8Array(tLbls);

  $("#test-pool-meta").textContent = `${state.testLabels.length} test samples`;
  $("#test-index-input").max = state.testLabels.length - 1;
  loadTestIndex(0);
}

function loadTestIndex(idx) {
  if (!state.testImages) return;
  const maxIdx = state.testLabels.length;
  state.testIndex = ((idx % maxIdx) + maxIdx) % maxIdx;
  $("#test-index-input").value = state.testIndex;
  const slice = state.testImages.slice(state.testIndex * LEN, (state.testIndex + 1) * LEN);
  const truth = state.testLabels[state.testIndex];
  $("#test-sample-meta").textContent = `Index #${state.testIndex} · True label: ${classLabel(truth)}`;
  setInputImage(slice);
}

function updateAccuracyChips() {
  for (const [m, acc] of Object.entries(state.accuracy)) {
    const el = $(`#acc-${m}`);
    if (el) el.textContent = `${(acc * 100).toFixed(1)}% (${(state.trainMs[m] / 1000).toFixed(1)}s)`;
  }
}

function renderPredictionResults(res) {
  // Centroid
  $("#pred-centroid").textContent = classLabel(res.centroid.prediction);
  const meansStrip = $("#means-strip");
  meansStrip.innerHTML = "";
  // Show top 10 relevant means or all if <= 10
  const maxMeans = Math.min(10, state.numClasses);
  for (let k = 0; k < maxMeans; k++) {
    const fig = document.createElement("figure");
    fig.className = "thumb" + (k === res.centroid.prediction ? " hit" : "");
    const c = document.createElement("canvas");
    c.width = c.height = 36;
    drawGray(c, state.modelData.centroid.means.slice(k * LEN, (k + 1) * LEN).map(Math.round));
    const cap = document.createElement("figcaption");
    cap.textContent = `${classLabel(k)}:${Math.round(res.centroid.distances[k])}`;
    fig.append(c, cap);
    meansStrip.append(fig);
  }

  // k-NN
  $("#pred-knn").textContent = classLabel(res.knn.prediction);
  const knnStrip = $("#knn-strip");
  knnStrip.innerHTML = "";
  res.knn.neighbors.forEach((n) => {
    const fig = document.createElement("figure");
    fig.className = "thumb" + (n.label === res.knn.prediction ? " hit" : "");
    const c = document.createElement("canvas");
    c.width = c.height = 36;
    const cap = document.createElement("figcaption");
    cap.textContent = `${classLabel(n.label)} (d=${Math.round(n.distance)})`;
    fig.append(c, cap);
    knnStrip.append(fig);
  });

  // Perceptron
  $("#pred-perceptron").textContent = classLabel(res.perceptron.prediction);
  renderBars($("#bars-perceptron"), res.perceptron.probs, res.perceptron.prediction);
  const wSlice = state.modelData.perceptron.weights.slice(
    res.perceptron.prediction * LEN,
    (res.perceptron.prediction + 1) * LEN
  );
  drawSigned($("#perceptron-heatmap"), wSlice);

  // MLP
  $("#pred-mlp").textContent = classLabel(res.mlp.prediction);
  renderBars($("#bars-mlp"), res.mlp.probs, res.mlp.prediction);
  const hiddenContainer = $("#mlp-hidden-activations");
  hiddenContainer.innerHTML = "";
  const maxAct = Math.max(...res.mlp.hidden, 1e-5);
  res.mlp.hidden.forEach((hVal, idx) => {
    const span = document.createElement("span");
    span.style.aspectRatio = "1";
    span.style.border = "1px solid var(--line)";
    const brightness = Math.round(255 - (hVal / maxAct) * 220);
    span.style.background = `rgb(${brightness},${brightness},${brightness})`;
    span.title = `Unit #${idx}: ${hVal.toFixed(3)}`;
    hiddenContainer.append(span);
  });

  // Consensus
  const preds = [
    res.centroid.prediction,
    res.knn.prediction,
    res.perceptron.prediction,
    res.mlp.prediction
  ];
  const freq = {};
  preds.forEach(p => freq[p] = (freq[p] || 0) + 1);
  const [topIdx, count] = Object.entries(freq).sort((a,b) => b[1] - a[1])[0];
  $("#consensus-text").textContent = `Consensus: ${count}/4 methods agree on '${classLabel(Number(topIdx))}'`;
}

function renderBars(container, probs, pred) {
  container.innerHTML = "";
  // Show top 6 candidates if many classes
  const indexed = probs.map((p, idx) => ({ p, idx }));
  indexed.sort((a, b) => b.p - a.p);
  const topList = indexed.slice(0, 8);

  topList.forEach(({ p, idx }) => {
    const row = document.createElement("div");
    row.className = "bar-row" + (idx === pred ? " hit" : "");
    row.innerHTML = `
      <span>${classLabel(idx)}</span>
      <div class="bar-track"><div class="bar-fill" style="width: ${(p * 100).toFixed(1)}%"></div></div>
      <span>${(p * 100).toFixed(1)}%</span>
    `;
    container.append(row);
  });
}

// Test sample selector
function setupTestSelector() {
  $("#prev-test-btn").onclick = () => loadTestIndex(state.testIndex - 1);
  $("#next-test-btn").onclick = () => loadTestIndex(state.testIndex + 1);
  $("#random-test-btn").onclick = () => loadTestIndex(Math.floor(Math.random() * (state.testLabels?.length || 1000)));
  $("#test-index-input").onchange = (e) => loadTestIndex(Number(e.target.value) || 0);

  const grid = $("#input-grid-canvas");
  grid.addEventListener("mousemove", (e) => {
    const r = grid.getBoundingClientRect();
    const x = Math.floor(((e.clientX - r.left) / r.width) * SIDE);
    const y = Math.floor(((e.clientY - r.top) / r.height) * SIDE);
    if (x >= 0 && x < SIDE && y >= 0 && y < SIDE) {
      const idx = y * SIDE + x;
      $("#grid-inspect-readout").textContent = `(x: ${x}, y: ${y}) · index: ${idx} · pixel: ${state.currentImage[idx]}`;
    }
  });
}

// Analytics & Charts
function renderCharts() {
  if (!state.baselineData) return;
  const canvas = $("#learning-curve-chart");
  const ctx = canvas.getContext("2d");
  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);

  const sizes = [10, 100, 1000, 10000, 60000];
  const methods = [
    { key: "centroid", label: "Centroid", color: "#10b981" },
    { key: "knn", label: "k-NN", color: "#6366f1" },
    { key: "perceptron", label: "Perceptron", color: "#f59e0b" },
    { key: "mlp", label: "MLP (ReLU)", color: "#ef4444" },
  ];

  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let acc = 60; acc <= 100; acc += 10) {
    const y = h - 30 - ((acc - 50) / 50) * (h - 50);
    ctx.moveTo(60, y);
    ctx.lineTo(w - 20, y);
    ctx.fillStyle = "#9ca3af";
    ctx.font = "11px monospace";
    ctx.fillText(`${acc}%`, 25, y + 4);
  }
  ctx.stroke();

  methods.forEach((m) => {
    ctx.strokeStyle = m.color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    sizes.forEach((sz, idx) => {
      const val = state.baselineData[String(sz)][m.key];
      const x = 70 + (idx / (sizes.length - 1)) * (w - 110);
      const y = h - 30 - ((val - 50) / 50) * (h - 50);
      idx === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    sizes.forEach((sz, idx) => {
      const val = state.baselineData[String(sz)][m.key];
      const x = 70 + (idx / (sizes.length - 1)) * (w - 110);
      const y = h - 30 - ((val - 50) / 50) * (h - 50);
      ctx.fillStyle = m.color;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  // Confusion Matrix
  if (state.confusion) {
    const table = $("#confusion-table");
    const mat = state.confusion.mlp;
    const n = Math.min(15, state.numClasses);
    let html = "<tr><th>Truth \\ Pred</th>" + Array.from({length: n}, (_, i) => `<th>${classLabel(i)}</th>`).join("") + "</tr>";
    for (let r = 0; r < n; r++) {
      html += `<tr><th>${classLabel(r)}</th>`;
      for (let c = 0; c < n; c++) {
        const count = mat[r][c];
        const cls = r === c ? "correct" : count > 0 ? "error" : "";
        html += `<td class="${cls}">${count}</td>`;
      }
      html += `</tr>`;
    }
    table.innerHTML = html;
  }

  // Disagreements gallery
  if (state.disagreements) {
    const gallery = $("#disagreements-gallery");
    gallery.innerHTML = "";
    state.disagreements.slice(0, 16).forEach((item) => {
      const card = document.createElement("div");
      card.style.border = "1px solid var(--line)";
      card.style.borderRadius = "6px";
      card.style.padding = "6px";
      card.style.background = "#fff";
      card.style.textAlign = "center";
      const c = document.createElement("canvas");
      c.width = c.height = 48;
      drawGray(c, item.image);
      card.append(c);
      const meta = document.createElement("div");
      meta.style.fontSize = "10px";
      meta.style.fontFamily = "monospace";
      meta.innerHTML = `Truth: <b>${classLabel(item.truth)}</b><br>MLP: ${classLabel(item.predictions.mlp)}<br>Cent: ${classLabel(item.predictions.centroid)}`;
      card.append(meta);
      card.onclick = () => {
        state.testIndex = item.index;
        $("#test-sample-meta").textContent = `Index #${item.index} · True label: ${classLabel(item.truth)}`;
        setInputImage(new Uint8Array(item.image));
        location.hash = "#/playground";
      };
      gallery.append(card);
    });
  }
}

// Code inspector
function setupCodeInspector() {
  const displayFile = (key) => {
    $$(".code-file-btn").forEach(b => b.classList.toggle("active", b.dataset.file === key));
    const file = rustFiles[key];
    if (!file) return;
    $("#code-rationale").textContent = `${file.title} — ${file.rationale}`;
    $("#code-display").textContent = file.code;
  };

  $$(".code-file-btn").forEach((btn) => {
    btn.onclick = () => displayFile(btn.dataset.file);
  });
  displayFile("centroid");
}

// Modals
function setupModals() {
  $$(".modal-trigger-btn").forEach((btn) => {
    btn.onclick = () => {
      const modal = $(`#modal-${btn.dataset.modal}`);
      if (modal) modal.showModal();
    };
  });
  $$(".close-modal-btn").forEach((btn) => {
    btn.onclick = () => btn.closest("dialog").close();
  });
}

// ONNX downloads
function setupOnnx() {
  $$(".onnx-dl-btn").forEach((btn) => {
    btn.onclick = () => {
      const method = btn.dataset.method;
      worker.postMessage({ type: "export_onnx", method });
    };
  });
}

function downloadBlob(bytes, filename) {
  const blob = new Blob([bytes], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Initial Boot: does NOT auto-train; initializes WASM and shows training modal
async function boot() {
  await init();
  worker.postMessage({ type: "init_wasm" });

  setupRouter();
  setupDrawing();
  setupTestSelector();
  setupTrainingModal();
  setupCodeInspector();
  setupModals();
  setupOnnx();

  // Load baseline JSON
  state.baselineData = await fetch("data/baseline.json").then(r => r.json());

  // Show training setup modal to let user choose configuration
  const modal = $("#modal-training-setup");
  modal.showModal();
}

boot().catch((err) => {
  const pill = $("#status-pill");
  pill.className = "status-pill error";
  pill.textContent = String(err);
});
