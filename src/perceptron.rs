//! Single-layer softmax classifier (multiclass perceptron).
//!
//! `x_i = pixel_i / 255`
//! `z_c = b_c + Σ_i w[c,i] · x_i`,  `p = softmax(z)`,  `L = −ln p_y`
//! `dL/dz_c = p_c − [c = y]`,  `dL/dw[c,i] = dL/dz_c · x_i`,  `dL/db_c = dL/dz_c`
//!
//! Mini-batch SGD; gradients are averaged over each batch.

use serde::Serialize;

use crate::image::{
    argmax, cross_entropy, dot, scale, scale_images, softmax, validate_training_set, Image,
    IMAGE_LEN, NUM_CLASSES,
};
use crate::rng::SplitMix64;

pub const LEARNING_RATE: f32 = 0.5;
pub const EPOCHS: u32 = 10;
pub const BATCH_SIZE: usize = 32;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Hyperparams {
    pub learning_rate: f32,
    pub epochs: u32,
    pub batch_size: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct PerceptronModel {
    /// `NUM_CLASSES * IMAGE_LEN`, class-major.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    /// Mean training loss after each epoch.
    pub loss_curve: Vec<f32>,
    pub hyperparams: Hyperparams,
}

#[derive(Clone, Debug, Serialize)]
pub struct PerceptronEvidence {
    pub prediction: u8,
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
}

fn logits(model: &PerceptronModel, x: &[f32]) -> Vec<f32> {
    (0..NUM_CLASSES)
        .map(|c| model.bias[c] + dot(&model.weights[c * IMAGE_LEN..(c + 1) * IMAGE_LEN], x))
        .collect()
}

pub fn train(images: &[Image], labels: &[u8], seed: u64) -> PerceptronModel {
    validate_training_set(images, labels);
    let x = scale_images(images);
    let mut model = PerceptronModel {
        weights: vec![0.0; NUM_CLASSES * IMAGE_LEN],
        bias: vec![0.0; NUM_CLASSES],
        loss_curve: Vec::with_capacity(EPOCHS as usize),
        hyperparams: Hyperparams {
            learning_rate: LEARNING_RATE,
            epochs: EPOCHS,
            batch_size: BATCH_SIZE as u32,
        },
    };
    let mut rng = SplitMix64::new(seed);
    let mut order: Vec<usize> = (0..images.len()).collect();
    let mut grad_w = vec![0.0f32; NUM_CLASSES * IMAGE_LEN];

    for _ in 0..EPOCHS {
        rng.shuffle(&mut order);
        for batch in order.chunks(BATCH_SIZE) {
            grad_w.fill(0.0);
            let mut grad_b = [0.0f32; NUM_CLASSES];
            let inv_n = 1.0 / batch.len() as f32;
            for &n in batch {
                let xn = &x[n * IMAGE_LEN..(n + 1) * IMAGE_LEN];
                let p = softmax(&logits(&model, xn));
                for c in 0..NUM_CLASSES {
                    let dz = (p[c] - if c == labels[n] as usize { 1.0 } else { 0.0 }) * inv_n;
                    grad_b[c] += dz;
                    for (g, &xi) in grad_w[c * IMAGE_LEN..(c + 1) * IMAGE_LEN]
                        .iter_mut()
                        .zip(xn)
                    {
                        *g += dz * xi;
                    }
                }
            }
            for (w, g) in model.weights.iter_mut().zip(&grad_w) {
                *w -= LEARNING_RATE * g;
            }
            for c in 0..NUM_CLASSES {
                model.bias[c] -= LEARNING_RATE * grad_b[c];
            }
        }
        let mut total = 0.0f64;
        for n in 0..images.len() {
            let p = softmax(&logits(&model, &x[n * IMAGE_LEN..(n + 1) * IMAGE_LEN]));
            total += cross_entropy(&p, labels[n]) as f64;
        }
        model.loss_curve.push((total / images.len() as f64) as f32);
    }
    model
}

pub fn predict(model: &PerceptronModel, image: &Image) -> PerceptronEvidence {
    let logits = logits(model, &scale(image.pixels()));
    let probs = softmax(&logits);
    PerceptronEvidence {
        prediction: argmax(&probs),
        logits,
        probs,
    }
}
