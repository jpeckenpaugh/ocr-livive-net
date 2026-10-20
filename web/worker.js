// Web Worker: Handles model training, testing, confusion matrix calculation, and ONNX export.
import init, { trainCentroid, trainKnn, trainPerceptron, trainMlp } from "../pkg/ocr.js";

const SEED = 0;
let models = {};
let currentTrainSize = 10000;
let datasets = {};

async function fetchBytes(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

function predictAll(image) {
  const out = {};
  for (const [name, model] of Object.entries(models)) {
    out[name] = model.predict(image);
  }
  return out;
}

async function loadDataset(size) {
  if (datasets[size]) return datasets[size];
  postMessage({ type: "status", text: `Downloading training set (${size} samples)…` });
  const [images, labels] = await Promise.all([
    fetchBytes(`../data/train-${size}.u8`),
    fetchBytes(`../data/train-${size}-labels.u8`),
  ]);
  datasets[size] = { images, labels };
  return datasets[size];
}

async function getTestSet(fullTest) {
  const testKey = fullTest ? "test-10000" : "test-1000";
  if (!datasets[testKey]) {
    postMessage({ type: "status", text: `Loading ${fullTest ? "10,000" : "1,000"} test samples…` });
    const [images, labels] = await Promise.all([
      fetchBytes(`../data/${testKey}.u8`),
      fetchBytes(`../data/${testKey}-labels.u8`),
    ]);
    datasets[testKey] = { images, labels };
  }
  return datasets[testKey];
}

async function retrain(size, fullTest = false) {
  currentTrainSize = size;
  const trainData = await loadDataset(size);
  const testData = await getTestSet(fullTest);

  models = {};
  const trainers = [
    ["centroid", trainCentroid],
    ["knn", trainKnn],
    ["perceptron", trainPerceptron],
    ["mlp", trainMlp],
  ];

  const modelData = {};
  const trainMs = {};

  for (const [name, trainFn] of trainers) {
    postMessage({ type: "status", text: `Training ${name} on ${size} samples…` });
    const t0 = performance.now();
    models[name] = trainFn(trainData.images, trainData.labels, SEED);
    trainMs[name] = performance.now() - t0;
    modelData[name] = models[name].modelData();
  }

  // Scoring & confusion matrix
  const testLen = testData.labels.length;
  // If k-NN on 60,000 and full 10,000 test, clamp k-NN evaluation to 1,000 to keep UI responsive
  const knnLimit = (size === 60000 && testLen > 1000) ? 1000 : testLen;
  postMessage({ type: "status", text: `Evaluating test accuracy on ${testLen} samples…` });

  const correct = { centroid: 0, knn: 0, perceptron: 0, mlp: 0 };
  const confusion = {
    centroid: Array.from({ length: 10 }, () => new Array(10).fill(0)),
    knn: Array.from({ length: 10 }, () => new Array(10).fill(0)),
    perceptron: Array.from({ length: 10 }, () => new Array(10).fill(0)),
    mlp: Array.from({ length: 10 }, () => new Array(10).fill(0)),
  };
  const disagreements = [];

  for (let i = 0; i < testLen; i++) {
    const imgSlice = testData.images.subarray(i * 784, (i + 1) * 784);
    const truth = testData.labels[i];

    const preds = {};
    for (const [mName, model] of Object.entries(models)) {
      if (mName === "knn" && i >= knnLimit) continue;
      const res = model.predict(imgSlice);
      preds[mName] = res.prediction;
      confusion[mName][truth][res.prediction]++;
      if (res.prediction === truth) correct[mName]++;
    }

    // Capture disagreements (where models disagree with each other or truth)
    if (disagreements.length < 50) {
      const predVals = Object.values(preds);
      const allSame = predVals.every(v => v === predVals[0]);
      if (!allSame || preds.mlp !== truth) {
        disagreements.push({
          index: i,
          truth,
          predictions: preds,
          image: Array.from(imgSlice),
        });
      }
    }
  }

  const accuracy = {
    centroid: correct.centroid / testLen,
    knn: correct.knn / (knnLimit),
    perceptron: correct.perceptron / testLen,
    mlp: correct.mlp / testLen,
  };

  postMessage({
    type: "ready",
    trainSize: size,
    modelData,
    accuracy,
    trainMs,
    confusion,
    disagreements,
    testCount: testLen,
  });
}

onmessage = async ({ data }) => {
  if (data.type === "predict") {
    postMessage({ type: "prediction", id: data.id, results: predictAll(data.image) });
  } else if (data.type === "retrain") {
    try {
      await retrain(data.size, data.fullTest);
    } catch (e) {
      postMessage({ type: "error", text: String(e) });
    }
  } else if (data.type === "export_onnx") {
    try {
      const model = models[data.method];
      if (!model) throw new Error(`Model ${data.method} not trained`);
      const bytes = model.toOnnx();
      postMessage({ type: "onnx_exported", method: data.method, bytes }, [bytes.buffer]);
    } catch (e) {
      postMessage({ type: "error", text: String(e) });
    }
  }
};

async function boot() {
  await init();
  await retrain(10000, false);
}

boot().catch((e) => postMessage({ type: "error", text: String(e) }));
