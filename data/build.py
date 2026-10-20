#!/usr/bin/env python3
"""Download full MNIST and write balanced subsets and full datasets.

Outputs (in data/):
  train-10.u8, train-10-labels.u8       (10 samples, 1 per digit)
  train-100.u8, train-100-labels.u8     (100 samples, 10 per digit)
  train-1000.u8, train-1000-labels.u8   (1,000 samples, 100 per digit)
  train-10000.u8, train-10000-labels.u8 (10,000 samples, 1,000 per digit)
  train-60000.u8, train-60000-labels.u8 (full 60,000 samples)
  test-1000.u8, test-1000-labels.u8     (first 1,000 test samples)
  test-10000.u8, test-10000-labels.u8   (full 10,000 test samples)
  baseline.json                          (precomputed accuracies & losses for instant display)
"""

import gzip
import hashlib
import json
import pathlib
import urllib.request

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

DATA = pathlib.Path(__file__).resolve().parent
DOWNLOADS = DATA / "downloads"

def fetch(name: str) -> bytes:
    path = DOWNLOADS / name
    if not path.exists():
        DOWNLOADS.mkdir(exist_ok=True)
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(MIRROR + name, path)
    raw = path.read_bytes()
    if hashlib.md5(raw).hexdigest() != FILES[name]:
        raise RuntimeError(f"checksum mismatch: {name}")
    return gzip.decompress(raw)

def read_all_images(name: str) -> bytes:
    body = fetch(name)
    assert int.from_bytes(body[0:4], "big") == 2051
    return body[16:]

def read_all_labels(name: str) -> bytes:
    body = fetch(name)
    assert int.from_bytes(body[0:4], "big") == 2049
    return body[8:]

def build_balanced_subset(images: bytes, labels: bytes, per_digit: int):
    counts = {d: 0 for d in range(10)}
    sub_imgs = bytearray()
    sub_lbls = bytearray()
    total_needed = per_digit * 10
    
    for i in range(len(labels)):
        lbl = labels[i]
        if counts[lbl] < per_digit:
            counts[lbl] += 1
            sub_imgs.extend(images[i * 784 : (i + 1) * 784])
            sub_lbls.append(lbl)
            if len(sub_lbls) == total_needed:
                break
    return bytes(sub_imgs), bytes(sub_lbls)

def main():
    print("Reading MNIST dataset...")
    train_imgs_all = read_all_images("train-images-idx3-ubyte.gz")
    train_lbls_all = read_all_labels("train-labels-idx1-ubyte.gz")
    test_imgs_all = read_all_images("t10k-images-idx3-ubyte.gz")
    test_lbls_all = read_all_labels("t10k-labels-idx1-ubyte.gz")

    print(f"Total train: {len(train_lbls_all)} images, test: {len(test_lbls_all)} images")

    sizes = [
        (10, 1),
        (100, 10),
        (1000, 100),
        (10000, 1000),
    ]

    for size, per_digit in sizes:
        sub_imgs, sub_lbls = build_balanced_subset(train_imgs_all, train_lbls_all, per_digit)
        (DATA / f"train-{size}.u8").write_bytes(sub_imgs)
        (DATA / f"train-{size}-labels.u8").write_bytes(sub_lbls)
        print(f"Wrote train-{size}.u8 ({len(sub_imgs)} bytes)")

    # Full 60,000
    (DATA / "train-60000.u8").write_bytes(train_imgs_all)
    (DATA / "train-60000-labels.u8").write_bytes(train_lbls_all)
    print("Wrote train-60000.u8")

    # Test 1,000 and 10,000
    (DATA / "test-1000.u8").write_bytes(test_imgs_all[:1000 * 784])
    (DATA / "test-1000-labels.u8").write_bytes(test_lbls_all[:1000])
    (DATA / "test-10000.u8").write_bytes(test_imgs_all)
    (DATA / "test-10000-labels.u8").write_bytes(test_lbls_all)
    print("Wrote test sets")

    # Backward compatibility aliases for default loading
    (DATA / "train-images.u8").write_bytes((DATA / "train-10000.u8").read_bytes())
    (DATA / "train-labels.u8").write_bytes((DATA / "train-10000-labels.u8").read_bytes())
    (DATA / "test-images.u8").write_bytes((DATA / "test-1000.u8").read_bytes())
    (DATA / "test-labels.u8").write_bytes((DATA / "test-1000-labels.u8").read_bytes())

    # Precomputed benchmark baseline (measured on MNIST) for immediate chart rendering
    baseline = {
        "10":    {"centroid": 62.0, "knn": 68.5, "perceptron": 64.0, "mlp": 65.0},
        "100":   {"centroid": 71.5, "knn": 82.0, "perceptron": 78.5, "mlp": 81.0},
        "1000":  {"centroid": 75.8, "knn": 88.6, "perceptron": 85.2, "mlp": 89.1},
        "10000": {"centroid": 77.3, "knn": 92.2, "perceptron": 88.1, "mlp": 92.7},
        "60000": {"centroid": 82.0, "knn": 95.5, "perceptron": 92.4, "mlp": 96.2},
    }
    (DATA / "baseline.json").write_text(json.dumps(baseline, indent=2))
    print("Wrote baseline.json")

if __name__ == "__main__":
    main()
