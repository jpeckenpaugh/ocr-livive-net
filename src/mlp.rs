//! One-hidden-layer MLP, 784 → 32 (ReLU) → 10 (softmax). Hand-written backprop.
//!
//! Forward:
//! `z1_h = b1_h + Σ_i w1[h,i] · x_i`,  `a_h = max(0, z1_h)`
//! `z2_c = b2_c + Σ_h w2[c,h] · a_h`,  `p = softmax(z2)`,  `L = −ln p_y`
//!
//! Backward (per image, divided by batch size N):
//! `dz2_c = (p_c − [c = y]) / N`
//! `dw2[c,h] = dz2_c · a_h`,  `db2_c = dz2_c`
//! `dz1_h = (Σ_c dz2_c · w2[c,h]) · [z1_h > 0]`
//! `dw1[h,i] = dz1_h · x_i`,  `db1_h = dz1_h`
//!
//! Init: He-uniform `±sqrt(6 / fan_in)`, zero biases.

use serde::Serialize;

use crate::image::{
    argmax, cross_entropy, dot, scale, scale_images, softmax, validate_training_set, Image,
    IMAGE_LEN, NUM_CLASSES,
};
use crate::rng::SplitMix64;

pub const HIDDEN: usize = 32;
pub const LEARNING_RATE: f32 = 0.1;
pub const EPOCHS: u32 = 15;
pub const BATCH_SIZE: usize = 32;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Hyperparams {
    pub hidden_units: u32,
    pub activation: &'static str,
    pub learning_rate: f32,
    pub epochs: u32,
    pub batch_size: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct MlpModel {
    /// `HIDDEN * IMAGE_LEN`, unit-major.
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// `NUM_CLASSES * HIDDEN`, class-major.
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub loss_curve: Vec<f32>,
    pub hyperparams: Hyperparams,
}

#[derive(Clone, Debug, Serialize)]
pub struct MlpEvidence {
    pub prediction: u8,
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
}

/// Returns `(z1, a, z2)`.
fn forward(model: &MlpModel, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let z1: Vec<f32> = (0..HIDDEN)
        .map(|h| model.b1[h] + dot(&model.w1[h * IMAGE_LEN..(h + 1) * IMAGE_LEN], x))
        .collect();
    let a: Vec<f32> = z1.iter().map(|&z| z.max(0.0)).collect();
    let z2: Vec<f32> = (0..NUM_CLASSES)
        .map(|c| model.b2[c] + dot(&model.w2[c * HIDDEN..(c + 1) * HIDDEN], &a))
        .collect();
    (z1, a, z2)
}

pub fn init(seed: u64) -> MlpModel {
    let mut rng = SplitMix64::new(seed);
    let l1 = (6.0 / IMAGE_LEN as f32).sqrt();
    let l2 = (6.0 / HIDDEN as f32).sqrt();
    MlpModel {
        w1: (0..HIDDEN * IMAGE_LEN).map(|_| rng.uniform(l1)).collect(),
        b1: vec![0.0; HIDDEN],
        w2: (0..NUM_CLASSES * HIDDEN).map(|_| rng.uniform(l2)).collect(),
        b2: vec![0.0; NUM_CLASSES],
        loss_curve: Vec::with_capacity(EPOCHS as usize),
        hyperparams: Hyperparams {
            hidden_units: HIDDEN as u32,
            activation: "relu",
            learning_rate: LEARNING_RATE,
            epochs: EPOCHS,
            batch_size: BATCH_SIZE as u32,
        },
    }
}

pub struct Gradient {
    pub loss: f32,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

/// Mean loss and gradient over `batch` (indices into scaled `x`).
pub fn gradient(model: &MlpModel, x: &[f32], labels: &[u8], batch: &[usize]) -> Gradient {
    let mut g = Gradient {
        loss: 0.0,
        w1: vec![0.0; HIDDEN * IMAGE_LEN],
        b1: vec![0.0; HIDDEN],
        w2: vec![0.0; NUM_CLASSES * HIDDEN],
        b2: vec![0.0; NUM_CLASSES],
    };
    let inv_n = 1.0 / batch.len() as f32;
    let mut loss = 0.0f64;
    for &n in batch {
        let xn = &x[n * IMAGE_LEN..(n + 1) * IMAGE_LEN];
        let (z1, a, z2) = forward(model, xn);
        let p = softmax(&z2);
        loss += cross_entropy(&p, labels[n]) as f64;

        let mut dz2 = [0.0f32; NUM_CLASSES];
        for c in 0..NUM_CLASSES {
            dz2[c] = (p[c] - if c == labels[n] as usize { 1.0 } else { 0.0 }) * inv_n;
            g.b2[c] += dz2[c];
            for h in 0..HIDDEN {
                g.w2[c * HIDDEN + h] += dz2[c] * a[h];
            }
        }
        for h in 0..HIDDEN {
            if z1[h] <= 0.0 {
                continue;
            }
            let mut dz1 = 0.0;
            for c in 0..NUM_CLASSES {
                dz1 += dz2[c] * model.w2[c * HIDDEN + h];
            }
            g.b1[h] += dz1;
            for (gw, &xi) in g.w1[h * IMAGE_LEN..(h + 1) * IMAGE_LEN].iter_mut().zip(xn) {
                *gw += dz1 * xi;
            }
        }
    }
    g.loss = (loss * inv_n as f64) as f32;
    g
}

fn step(params: &mut [f32], grads: &[f32]) {
    for (p, g) in params.iter_mut().zip(grads) {
        *p -= LEARNING_RATE * g;
    }
}

pub fn train(images: &[Image], labels: &[u8], seed: u64) -> MlpModel {
    validate_training_set(images, labels);
    let x = scale_images(images);
    let mut model = init(seed);
    let mut rng = SplitMix64::new(seed ^ 0x5EED);
    let mut order: Vec<usize> = (0..images.len()).collect();

    for _ in 0..EPOCHS {
        rng.shuffle(&mut order);
        for batch in order.chunks(BATCH_SIZE) {
            let g = gradient(&model, &x, labels, batch);
            step(&mut model.w1, &g.w1);
            step(&mut model.b1, &g.b1);
            step(&mut model.w2, &g.w2);
            step(&mut model.b2, &g.b2);
        }
        let mut total = 0.0f64;
        for n in 0..images.len() {
            let (_, _, z2) = forward(&model, &x[n * IMAGE_LEN..(n + 1) * IMAGE_LEN]);
            total += cross_entropy(&softmax(&z2), labels[n]) as f64;
        }
        model.loss_curve.push((total / images.len() as f64) as f32);
    }
    model
}

pub fn predict(model: &MlpModel, image: &Image) -> MlpEvidence {
    let (_, hidden, logits) = forward(model, &scale(image.pixels()));
    let probs = softmax(&logits);
    MlpEvidence {
        prediction: argmax(&probs),
        hidden,
        logits,
        probs,
    }
}
