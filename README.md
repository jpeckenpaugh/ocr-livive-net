# ocr-livive-net

Browser-native OCR from scratch in WebAssembly. Four transparent classifiers (Centroid, k-NN, Perceptron, hand-written MLP), zero ML dependencies, and paper-and-pencil auditable arithmetic.

Live at <https://ocr.livive.net>.

## Classifiers

All train in the browser on 10,000 MNIST images (28×28) and are scored on 1,000 test images.

| Method | Model | Test accuracy |
|---|---|---|
| Centroid | 10 per-digit mean images; nearest by Euclidean distance | ~77% |
| k-NN (k=3) | the training set; 3 nearest vote | ~92% |
| Perceptron | 10×784 weights + softmax, mini-batch SGD | ~88% |
| MLP | 784→32 (ReLU)→10 softmax, hand-written backprop | ~93% |

Drawn digits go through `src/preprocess.rs`: integer grayscale, bounding box, box filter to 20×20, center of mass placed at the center of the 28×28 frame (MNIST's own normalization).

## Layout

```text
src/        Rust crate: image.rs, preprocess.rs, centroid.rs, knn.rs, perceptron.rs, mlp.rs, lib.rs (wasm boundary)
tests/      cargo tests
examples/   eval.rs: native train + test accuracy
data/       build.py and the MNIST subset as raw bytes (*.u8)
web/        app.js, worker.js (training off the main thread), styles.css
scripts/    build-web.sh, serve.sh
.github/    CI: test, build wasm, deploy to GitHub Pages
```

## Develop

Requires Rust 1.99.1 (pinned in `rust-toolchain.toml`) and `wasm-bindgen-cli` 0.2.129:

```bash
cargo install wasm-bindgen-cli --version 0.2.129 --locked

cargo test --release
cargo run --release --example eval   # native accuracy check
./scripts/build-web.sh               # -> pkg/
./scripts/serve.sh 8000              # http://localhost:8000/
```

Rebuild the dataset (already committed) with `python3 data/build.py`.

## Deploy

`.github/workflows/pages.yml` runs the tests, builds `pkg/`, and deploys to GitHub Pages on every push to `main`. In the repo settings, set Pages → Source to **GitHub Actions**.
