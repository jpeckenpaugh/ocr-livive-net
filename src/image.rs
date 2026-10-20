//! Shared types and arithmetic.
//!
//! An `Image` is 784 bytes, row-major 28x28, 0 = background, 255 = ink
//! (MNIST convention). Index of `(x, y)` is `y * 28 + x`.
//! Supports arbitrary class counts (e.g. 10 for digits, 26 for letters, 36 for alphanumeric).

use serde::Serialize;

pub const IMAGE_SIDE: usize = 28;
pub const IMAGE_LEN: usize = IMAGE_SIDE * IMAGE_SIDE;
pub const DEFAULT_SEED: u64 = 0;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct Image {
    pixels: Vec<u8>,
}

impl Image {
    pub fn new(pixels: Vec<u8>) -> Self {
        assert_eq!(pixels.len(), IMAGE_LEN, "image must be {IMAGE_LEN} bytes");
        Self { pixels }
    }

    pub fn from_slice(pixels: &[u8]) -> Self {
        Self::new(pixels.to_vec())
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub fn into_pixels(self) -> Vec<u8> {
        self.pixels
    }
}

/// Split a flat `n * 784` buffer into images.
pub fn split_images(flat: &[u8]) -> Vec<Image> {
    assert_eq!(
        flat.len() % IMAGE_LEN,
        0,
        "flat buffer is not a multiple of {IMAGE_LEN}"
    );
    flat.chunks_exact(IMAGE_LEN)
        .map(Image::from_slice)
        .collect()
}

pub fn infer_num_classes(labels: &[u8]) -> usize {
    assert!(!labels.is_empty(), "labels slice is empty");
    (labels.iter().copied().max().unwrap_or(0) as usize) + 1
}

pub fn validate_training_set(images: &[Image], labels: &[u8]) {
    assert!(!images.is_empty(), "training set is empty");
    assert_eq!(images.len(), labels.len(), "image/label count mismatch");
}

pub fn class_counts(labels: &[u8], num_classes: usize) -> Vec<usize> {
    let mut counts = vec![0; num_classes];
    for &l in labels {
        counts[l as usize] += 1;
    }
    counts
}

/// Index of the largest value; ties go to the lowest index.
pub fn argmax(values: &[f32]) -> u8 {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best as u8
}

/// Numerically stable softmax (subtract the max logit first).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Cross-entropy of the true class, with p clamped away from 0.
pub fn cross_entropy(probs: &[f32], label: u8) -> f32 {
    -probs[label as usize].max(1e-7).ln()
}

/// Squared Euclidean distance between two byte images.
pub fn squared_distance(a: &[u8], b: &[u8]) -> u32 {
    let mut sum = 0u32;
    for i in 0..a.len() {
        let d = a[i] as i32 - b[i] as i32;
        sum += (d * d) as u32;
    }
    sum
}

/// Eight-accumulator dot product (independent chains vectorize better).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let octets = a.len() / 8;
    for o in 0..octets {
        let i = o * 8;
        for k in 0..8 {
            acc[k] += a[i + k] * b[i + k];
        }
    }
    let mut sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for i in octets * 8..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Scale bytes to 0.0..=1.0 into one contiguous matrix.
pub fn scale_images(images: &[Image]) -> Vec<f32> {
    let mut out = Vec::with_capacity(images.len() * IMAGE_LEN);
    for image in images {
        out.extend(image.pixels().iter().map(|&p| p as f32 / 255.0));
    }
    out
}

pub fn scale(pixels: &[u8]) -> Vec<f32> {
    pixels.iter().map(|&p| p as f32 / 255.0).collect()
}
