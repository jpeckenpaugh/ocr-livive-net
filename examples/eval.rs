//! Train all four classifiers on data/train-* and report accuracy on data/test-*.
//! Run: `cargo run --release --example eval`

use std::time::Instant;

use ocr::image::{split_images, IMAGE_LEN};
use ocr::{centroid, knn, mlp, perceptron};

fn read(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/data/{name}", env!("CARGO_MANIFEST_DIR")))
        .unwrap_or_else(|e| panic!("{name}: {e} (run data/build.py)"))
}

fn report(name: &str, start: Instant, labels: &[u8], predict: impl Fn(usize) -> u8) {
    let correct = (0..labels.len())
        .filter(|&i| predict(i) == labels[i])
        .count();
    println!(
        "{name:<11} {:>6.2}%  ({:.1}s)",
        100.0 * correct as f64 / labels.len() as f64,
        start.elapsed().as_secs_f64()
    );
}

fn main() {
    let train = split_images(&read("train-images.u8"));
    let train_labels = read("train-labels.u8");
    let test = split_images(&read("test-images.u8"));
    let test_labels = read("test-labels.u8");
    assert_eq!(train[0].pixels().len(), IMAGE_LEN);
    println!("train {} / test {}", train.len(), test.len());

    let t = Instant::now();
    let m = centroid::train(&train, &train_labels);
    report("centroid", t, &test_labels, |i| {
        centroid::predict(&m, &test[i]).prediction
    });

    let t = Instant::now();
    let m = knn::train(&train, &train_labels);
    report("knn", t, &test_labels, |i| {
        knn::predict(&m, &test[i]).prediction
    });

    let t = Instant::now();
    let m = perceptron::train(&train, &train_labels, 0);
    report("perceptron", t, &test_labels, |i| {
        perceptron::predict(&m, &test[i]).prediction
    });

    let t = Instant::now();
    let m = mlp::train(&train, &train_labels, 0);
    report("mlp", t, &test_labels, |i| {
        mlp::predict(&m, &test[i]).prediction
    });
}
