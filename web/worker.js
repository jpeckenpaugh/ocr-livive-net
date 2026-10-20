// Web Worker: Handles model training, testing, confusion matrix calculation, and ONNX export.
import init, { trainCentroid, trainKnn, trainPerceptron, trainMlp } from "../pkg/ocr.js";

const SEED = 0;
let models = {};
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

async function loadDataset(datasetType, size) {
  const key = `${datasetType}-${size}`;
  if (datasets[key]) return datasets[key];
  postMessage({ type: "progress", step: "download", text: `Downloading ${datasetType} training data (${size} samples)…`, pct: 15 });
  
  let imgPath, lblPath;
  if (datasetType === "digits") {
    imgPath = `../data/train-${size}.u8`;
    lblPath = `../data/train-${size}-labels.u8`;
  } else if (datasetType === "letters") {
    imgPath = `../data/train-letters-${size}.u8`;
    lblPath = `../data/train-letters-${size}-labels.u8`;
  } else {
    // Alphanumeric
    imgPath = `../data/train-alphanumeric-${size}.u8`;
    lblPath = `../data/train-alphanumeric-${size}-labels.u8`;
  }

  const [images, labels] = await Promise.all([
    fetchBytes(imgPath),
    fetchBytes(lblPath),
  ]);
  datasets[key] = { images, labels };
  return datasets[key];
}

async function getTestSet(datasetType, fullTest) {
  const testKey = `${datasetType}-${fullTest ? "10k" : "1k"}`;
  if (!datasets[testKey]) {
    postMessage({ type: "progress", step: "download_test", text: `Loading test evaluation set…`, pct: 30 });
    let imgPath, lblPath;
    if (datasetType === "digits") {
      imgPath = fullTest ? "../data/test-10000.u8" : "../data/test-1000.u8";
      lblPath = fullTest ? "../data/test-10000-labels.u8" : "../data/test-1000-labels.u8";
    } else if (datasetType === "letters") {
      imgPath = "../data/test-letters-1000.u8";
      lblPath = "../data/test-letters-1000-labels.u8";
    } else {
      imgPath = "../data/test-alphanumeric-1000.u8";
      lblPath = "../data/test-alphanumeric-1000-labels.u8";
    }
    const [images, labels] = await Promise.all([
      fetchBytes(imgPath),
      fetchBytes(lblPath),
    ]);
    datasets[testKey] = { images, labels };
  }
  return datasets[testKey];
}

async function retrain(datasetType, size, fullTest = false) {
  const trainData = await loadDataset(datasetType, size);
  const testData = await getTestSet(datasetType, fullTest);

  models = {};
  const trainers = [
    ["centroid", trainCentroid, 45],
    ["knn", trainKnn, 60],
    ["perceptron", trainPerceptron, 75],
    ["mlp", trainMlp, 90],
  ];

  const modelData = {};
  const trainMs = {};

  for (const [name, trainFn, pct] of trainers) {
    postMessage({ type: "progress", step: `train_${name}`, text: `Training ${name} on ${trainData.labels.length} samples…`, pct });
    const t0 = performance.now();
    models[name] = trainFn(trainData.images, trainData.labels, SEED);
    trainMs[name] = performance.now() - t0;
    modelData[name] = models[name].modelData();
  }

  const testLen = testData.labels.length;
  const numClasses = datasetType === "digits" ? 10 : datasetType === "letters" ? 26 : 36;
  postMessage({ type: "progress", step: "scoring", text: `Evaluating test set (${testLen} samples)…`, pct: 95 });

  const correct = { centroid: 0, knn: 0, perceptron: 0, mlp: 0 };
  const confusion = {
    centroid: Array.from({ length: numClasses }, () => new Array(numClasses).fill(0)),
    knn: Array.from({ length: numClasses }, () => new Array(numClasses).fill(0)),
    perceptron: Array.from({ length: numClasses }, () => new Array(numClasses).fill(0)),
    mlp: Array.from({ length: numClasses }, () => new Array(numClasses).fill(0)),
  };
  const disagreements = [];

  for (let i = 0; i < testLen; i++) {
    const imgSlice = testData.images.subarray(i * 784, (i + 1) * 784);
    const truth = testData.labels[i];

    const preds = {};
    for (const [mName, model] of Object.entries(models)) {
      const res = model.predict(imgSlice);
      preds[mName] = res.prediction;
      if (truth < numClasses && res.prediction < numClasses) {
        confusion[mName][truth][res.prediction]++;
      }
      if (res.prediction === truth) correct[mName]++;
    }

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
    knn: correct.knn / testLen,
    perceptron: correct.perceptron / testLen,
    mlp: correct.mlp / testLen,
  };

  postMessage({
    type: "ready",
    datasetType,
    trainSize: size,
    numClasses,
    actualSamples: trainData.labels.length,
    modelData,
    accuracy,
    trainMs,
    confusion,
    disagreements,
    testCount: testLen,
  });
}

onmessage = async ({ data }) => {
  if (data.type === "init_wasm") {
    await init();
    postMessage({ type: "wasm_initialized" });
  } else if (data.type === "predict") {
    postMessage({ type: "prediction", id: data.id, results: predictAll(data.image) });
  } else if (data.type === "retrain") {
    try {
      await retrain(data.datasetType, data.size, data.fullTest);
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
