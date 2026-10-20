//! k-nearest neighbors. The model is the training set.
//!
//! Rank training images by Euclidean distance, take the `K` nearest, vote.
//! Vote ties go to the tied label whose neighbor is nearest.

use serde::Serialize;

use crate::image::{infer_num_classes, squared_distance, validate_training_set, Image, IMAGE_LEN};

pub const K: usize = 3;

#[derive(Clone, Debug)]
pub struct KnnModel {
    pub num_classes: usize,
    /// Flat `n * IMAGE_LEN`.
    pub images: Vec<u8>,
    pub labels: Vec<u8>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Neighbor {
    /// Index into the training set.
    pub index: usize,
    pub label: u8,
    pub distance: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct KnnEvidence {
    pub prediction: u8,
    pub votes: Vec<u8>,
    pub neighbors: Vec<Neighbor>,
}

pub fn train(images: &[Image], labels: &[u8]) -> KnnModel {
    validate_training_set(images, labels);
    assert!(images.len() >= K, "need at least {K} training images");
    let num_classes = infer_num_classes(labels);
    let mut flat = Vec::with_capacity(images.len() * IMAGE_LEN);
    for image in images {
        flat.extend_from_slice(image.pixels());
    }
    KnnModel {
        num_classes,
        images: flat,
        labels: labels.to_vec(),
    }
}

pub fn predict(model: &KnnModel, image: &Image) -> KnnEvidence {
    let mut nearest: Vec<(u32, usize)> = Vec::with_capacity(K + 1);
    for (index, train) in model.images.chunks_exact(IMAGE_LEN).enumerate() {
        let d = squared_distance(image.pixels(), train);
        if nearest.len() == K && d >= nearest[K - 1].0 {
            continue;
        }
        let pos = nearest
            .iter()
            .position(|&(nd, _)| d < nd)
            .unwrap_or(nearest.len());
        nearest.insert(pos, (d, index));
        nearest.truncate(K);
    }

    let mut votes = vec![0u8; model.num_classes];
    let neighbors: Vec<Neighbor> = nearest
        .iter()
        .map(|&(d, index)| {
            let label = model.labels[index];
            votes[label as usize] += 1;
            Neighbor {
                index,
                label,
                distance: (d as f64).sqrt(),
            }
        })
        .collect();

    let top = *votes.iter().max().unwrap();
    let prediction = neighbors
        .iter()
        .find(|n| votes[n.label as usize] == top)
        .unwrap()
        .label;

    KnnEvidence {
        prediction,
        votes,
        neighbors,
    }
}
