#!/usr/bin/env python3
"""Download EMNIST Letters (Public Domain / CC0 from NIST), orient upright,
and extract balanced subsets for A-Z letters and 0-9+A-Z alphanumeric.
"""

import gzip
import pathlib
import ssl
import urllib.request

DATA = pathlib.Path(__file__).resolve().parent
DOWNLOADS = DATA / "downloads"

HF_BASE = "https://huggingface.co/datasets/Royc30ne/emnist-letters/resolve/main/"
FILES = [
    "emnist-letters-train-images-idx3-ubyte.gz",
    "emnist-letters-train-labels-idx1-ubyte.gz",
    "emnist-letters-test-images-idx3-ubyte.gz",
    "emnist-letters-test-labels-idx1-ubyte.gz",
]

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def fetch(name: str) -> bytes:
    dest = DOWNLOADS / name
    if not dest.exists():
        DOWNLOADS.mkdir(exist_ok=True)
        print(f"Downloading {name}...")
        req = urllib.request.Request(HF_BASE + name, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as resp:
            data = resp.read()
            dest.write_bytes(data)
    return gzip.decompress(dest.read_bytes())

def orient_emnist_image(raw_bytes_784: bytes) -> bytes:
    """EMNIST raw images are transposed (column-major) relative to MNIST (row-major).
    Transposing orientates the glyphs upright."""
    out = bytearray(784)
    for y in range(28):
        for x in range(28):
            out[y * 28 + x] = raw_bytes_784[x * 28 + y]
    return bytes(out)

def read_all_images(name: str) -> list:
    body = fetch(name)
    assert int.from_bytes(body[0:4], "big") == 2051
    num = int.from_bytes(body[4:8], "big")
    print(f"Loading {num} raw images from {name}...")
    images = []
    raw = body[16:]
    for i in range(num):
        img_slice = raw[i * 784 : (i + 1) * 784]
        images.append(orient_emnist_image(img_slice))
    return images

def read_all_labels(name: str) -> list:
    body = fetch(name)
    assert int.from_bytes(body[0:4], "big") == 2049
    num = int.from_bytes(body[4:8], "big")
    # EMNIST letters labels are 1..26 -> normalize to 0..25 (0 = 'A', 25 = 'Z')
    return [l - 1 for l in body[8 : 8 + num]]

def main():
    train_imgs = read_all_images("emnist-letters-train-images-idx3-ubyte.gz")
    train_lbls = read_all_labels("emnist-letters-train-labels-idx1-ubyte.gz")
    test_imgs = read_all_images("emnist-letters-test-images-idx3-ubyte.gz")
    test_lbls = read_all_labels("emnist-letters-test-labels-idx1-ubyte.gz")

    print(f"Loaded {len(train_imgs)} train letters, {len(test_imgs)} test letters.")

    # Subsets for Letters (26 classes):
    # - letters-100: ~4 per letter (total 104)
    # - letters-1000: ~38 per letter (total 988)
    # - letters-10000: ~384 per letter (total 9,984)
    subsets = [
        ("letters-100", 4),
        ("letters-1000", 38),
        ("letters-10000", 384),
    ]

    for name, per_class in subsets:
        counts = {c: 0 for c in range(26)}
        sub_i = bytearray()
        sub_l = bytearray()
        for img, lbl in zip(train_imgs, train_lbls):
            if counts[lbl] < per_class:
                counts[lbl] += 1
                sub_i.extend(img)
                sub_l.append(lbl)
        (DATA / f"train-{name}.u8").write_bytes(sub_i)
        (DATA / f"train-{name}-labels.u8").write_bytes(sub_l)
        print(f"Wrote {name}: {len(sub_l)} samples")

    # Letters Test Set (1,040 samples = 40 per letter)
    test_counts = {c: 0 for c in range(26)}
    t_imgs = bytearray()
    t_lbls = bytearray()
    for img, lbl in zip(test_imgs, test_lbls):
        if test_counts[lbl] < 40:
            test_counts[lbl] += 1
            t_imgs.extend(img)
            t_lbls.append(lbl)
            if len(t_lbls) == 1040:
                break
    (DATA / "test-letters-1000.u8").write_bytes(t_imgs)
    (DATA / "test-letters-1000-labels.u8").write_bytes(t_lbls)
    print("Wrote test-letters-1000.u8 (1,040 samples)")

if __name__ == "__main__":
    main()

# Alphanumeric dataset builder: combines MNIST digits (0-9) and EMNIST Letters (A-Z) -> 36 classes
def build_alphanumeric():
    d_train_img = (DATA / "train-10000.u8").read_bytes()
    d_train_lbl = (DATA / "train-10000-labels.u8").read_bytes()
    l_train_img = (DATA / "train-letters-10000.u8").read_bytes()
    l_train_lbl = (DATA / "train-letters-10000-labels.u8").read_bytes()

    # Digits are labels 0..9, Letters offset to 10..35
    combo_img = bytearray(d_train_img)
    combo_lbl = bytearray(d_train_lbl)
    combo_img.extend(l_train_img)
    for lbl in l_train_lbl:
        combo_lbl.append(lbl + 10)

    (DATA / "train-alphanumeric-10000.u8").write_bytes(combo_img)
    (DATA / "train-alphanumeric-10000-labels.u8").write_bytes(combo_lbl)
    print(f"Wrote train-alphanumeric-10000: {len(combo_lbl)} samples (36 classes)")

    d_test_img = (DATA / "test-1000.u8").read_bytes()
    d_test_lbl = (DATA / "test-1000-labels.u8").read_bytes()
    l_test_img = (DATA / "test-letters-1000.u8").read_bytes()
    l_test_lbl = (DATA / "test-letters-1000-labels.u8").read_bytes()

    combo_test_img = bytearray(d_test_img)
    combo_test_lbl = bytearray(d_test_lbl)
    combo_test_img.extend(l_test_img)
    for lbl in l_test_lbl:
        combo_test_lbl.append(lbl + 10)

    (DATA / "test-alphanumeric-1000.u8").write_bytes(combo_test_img)
    (DATA / "test-alphanumeric-1000-labels.u8").write_bytes(combo_test_lbl)
    print(f"Wrote test-alphanumeric-1000: {len(combo_test_lbl)} samples (36 classes)")

build_alphanumeric()
