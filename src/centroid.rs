//! Nearest class mean.
//!
//! `mean_c[i]` = average of pixel `i` over class `c`.
//! `d_c = sqrt(Σ (x_i − mean_c[i])²)`; prediction = argmin `d_c`.

use serde::Serialize;

use crate::image::{class_counts, infer_num_classes, validate_training_set, Image, IMAGE_LEN};

#[derive(Clone, Debug, Serialize)]
pub struct CentroidModel {
    pub num_classes: usize,
    /// `num_classes * IMAGE_LEN`, class-major.
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
    let num_classes = infer_num_classes(labels);
    let counts = class_counts(labels, num_classes);
    assert!(
        counts.iter().all(|&c| c > 0),
        "every class needs at least one image"
    );

    let mut sums = vec![0.0f64; num_classes * IMAGE_LEN];
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
        num_classes,
        means,
        counts,
    }
}

pub fn predict(model: &CentroidModel, image: &Image) -> CentroidEvidence {
    let pixels = image.pixels();
    let num_classes = model.num_classes;
    let mut distances = Vec::with_capacity(num_classes);
    for c in 0..num_classes {
        let mean = &model.means[c * IMAGE_LEN..(c + 1) * IMAGE_LEN];
        let mut sum = 0.0f64;
        for i in 0..IMAGE_LEN {
            let d = pixels[i] as f64 - mean[i] as f64;
            sum += d * d;
        }
        distances.push(sum.sqrt());
    }
    let mut best = 0;
    for c in 1..num_classes {
        if distances[c] < distances[best] {
            best = c;
        }
    }
    CentroidEvidence {
        prediction: best as u8,
        distances,
    }
}
