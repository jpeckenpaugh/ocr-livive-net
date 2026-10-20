//! Nearest class mean.
//!
//! `mean_c[i]` = average of pixel `i` over class `c`.
//! `d_c = sqrt(Σ (x_i − mean_c[i])²)`; prediction = argmin `d_c`.

use serde::Serialize;

use crate::image::{class_counts, validate_training_set, Image, IMAGE_LEN, NUM_CLASSES};

#[derive(Clone, Debug, Serialize)]
pub struct CentroidModel {
    /// `NUM_CLASSES * IMAGE_LEN`, class-major.
    pub means: Vec<f32>,
    pub counts: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CentroidEvidence {
    pub prediction: u8,
    pub distances: Vec<f64>,
}

pub fn train(images: &[Image], labels: &[u8]) -> CentroidModel {
    validate_training_set(images, labels);
    let counts = class_counts(labels);
    assert!(
        counts.iter().all(|&c| c > 0),
        "every digit needs at least one image"
    );

    let mut sums = vec![0.0f64; NUM_CLASSES * IMAGE_LEN];
    for (image, &label) in images.iter().zip(labels) {
        let row = &mut sums[label as usize * IMAGE_LEN..(label as usize + 1) * IMAGE_LEN];
        for (s, &p) in row.iter_mut().zip(image.pixels()) {
            *s += p as f64;
        }
    }
    let means = sums
        .iter()
        .enumerate()
        .map(|(i, &s)| (s / counts[i / IMAGE_LEN] as f64) as f32)
        .collect();

    CentroidModel {
        means,
        counts: counts.to_vec(),
    }
}

pub fn predict(model: &CentroidModel, image: &Image) -> CentroidEvidence {
    let pixels = image.pixels();
    let mut distances = Vec::with_capacity(NUM_CLASSES);
    for c in 0..NUM_CLASSES {
        let mean = &model.means[c * IMAGE_LEN..(c + 1) * IMAGE_LEN];
        let mut sum = 0.0f64;
        for i in 0..IMAGE_LEN {
            let d = pixels[i] as f64 - mean[i] as f64;
            sum += d * d;
        }
        distances.push(sum.sqrt());
    }
    // Strict `<` keeps the lowest digit on an exact tie.
    let mut best = 0;
    for c in 1..NUM_CLASSES {
        if distances[c] < distances[best] {
            best = c;
        }
    }
    CentroidEvidence {
        prediction: best as u8,
        distances,
    }
}
