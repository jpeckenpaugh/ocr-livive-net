use ocr::image::{argmax, softmax, Image, IMAGE_LEN, IMAGE_SIDE, NUM_CLASSES};
use ocr::{centroid, knn, mlp, perceptron, preprocess};

/// Ten synthetic digits: class `c` lights row band `c` (rows 2c..2c+2).
fn synthetic() -> (Vec<Image>, Vec<u8>) {
    let mut images = Vec::new();
    let mut labels = Vec::new();
    for rep in 0..4u8 {
        for c in 0..NUM_CLASSES {
            let mut px = vec![rep * 5; IMAGE_LEN];
            for y in 2 * c + 4..2 * c + 6 {
                for x in 4..24 {
                    px[y * IMAGE_SIDE + x] = 255 - rep * 10;
                }
            }
            images.push(Image::new(px));
            labels.push(c as u8);
        }
    }
    (images, labels)
}

#[test]
fn softmax_sums_to_one_and_argmax_ties_low() {
    let p = softmax(&[1.0, 2.0, 3.0]);
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert_eq!(argmax(&[0.5, 0.9, 0.9]), 1);
}

#[test]
fn centroid_means_and_prediction() {
    let (images, labels) = synthetic();
    let model = centroid::train(&images, &labels);
    // Background pixel mean for class 0 = (0 + 5 + 10 + 15) / 4.
    assert_eq!(model.means[0], 7.5);
    for (img, &l) in images.iter().zip(&labels) {
        assert_eq!(centroid::predict(&model, img).prediction, l);
    }
}

#[test]
fn knn_votes_and_neighbors() {
    let (images, labels) = synthetic();
    let model = knn::train(&images, &labels);
    let ev = knn::predict(&model, &images[3]);
    assert_eq!(ev.prediction, 3);
    assert_eq!(ev.neighbors.len(), knn::K);
    assert_eq!(ev.neighbors[0].distance, 0.0);
    assert_eq!(ev.votes.iter().map(|&v| v as usize).sum::<usize>(), knn::K);
}

#[test]
fn perceptron_and_mlp_fit_synthetic_set() {
    let (images, labels) = synthetic();
    let p = perceptron::train(&images, &labels, 0);
    let m = mlp::train(&images, &labels, 0);
    assert!(p.loss_curve.last() < p.loss_curve.first());
    assert!(m.loss_curve.last() < m.loss_curve.first());
    for (img, &l) in images.iter().zip(&labels) {
        assert_eq!(perceptron::predict(&p, img).prediction, l);
        assert_eq!(mlp::predict(&m, img).prediction, l);
    }
}

#[test]
fn mlp_gradient_matches_finite_difference() {
    let (images, labels) = synthetic();
    let x = ocr::image::scale_images(&images);
    let model = mlp::init(1);
    let batch: Vec<usize> = (0..images.len()).collect();
    let g = mlp::gradient(&model, &x, &labels, &batch);
    let eps = 1e-2;
    for &i in &[0usize, 5, 17] {
        let mut plus = model.clone();
        plus.w2[i] += eps;
        let mut minus = model.clone();
        minus.w2[i] -= eps;
        let lp = mlp::gradient(&plus, &x, &labels, &batch).loss;
        let lm = mlp::gradient(&minus, &x, &labels, &batch).loss;
        let numeric = (lp - lm) / (2.0 * eps);
        assert!(
            (numeric - g.w2[i]).abs() < 1e-3,
            "w2[{i}]: {numeric} vs {}",
            g.w2[i]
        );
    }
}

#[test]
fn preprocess_grayscale_formula() {
    assert_eq!(preprocess::grayscale(&[255, 0, 0, 255], 1, 1), vec![76]);
    assert_eq!(preprocess::grayscale(&[128, 128, 128, 0], 1, 1), vec![128]);
}

#[test]
fn preprocess_blank_and_centered_square() {
    let (w, h) = (100usize, 100usize);
    let blank = vec![0u8; w * h * 4];
    assert!(preprocess::preprocess(&blank, 100, 100)
        .pixels()
        .iter()
        .all(|&p| p == 0));

    // White 40x40 square in the top-left corner → 20x20 block centered in 28x28.
    let mut rgba = blank;
    for y in 0..40 {
        for x in 0..40 {
            rgba[(y * w + x) * 4..(y * w + x) * 4 + 3].copy_from_slice(&[255, 255, 255]);
        }
    }
    let img = preprocess::preprocess(&rgba, 100, 100);
    let px = img.pixels();
    assert_eq!(px[4 * IMAGE_SIDE + 4], 255);
    assert_eq!(px[23 * IMAGE_SIDE + 23], 255);
    assert_eq!(px[3 * IMAGE_SIDE + 3], 0);
    assert_eq!(px[24 * IMAGE_SIDE + 24], 0);
}

#[test]
fn onnx_export_produces_non_empty_protobuf() {
    let (images, labels) = synthetic();
    let c = centroid::train(&images, &labels);
    let p = perceptron::train(&images, &labels, 0);
    let m = mlp::train(&images, &labels, 0);
    let k = knn::train(&images, &labels);

    let onnx_c = ocr::onnx::export_centroid(&c);
    let onnx_p = ocr::onnx::export_perceptron(&p);
    let onnx_m = ocr::onnx::export_mlp(&m);
    let onnx_k = ocr::onnx::export_knn(&k);

    assert!(onnx_c.len() > 100);
    assert!(onnx_p.len() > 100);
    assert!(onnx_m.len() > 100);
    assert!(onnx_k.len() > 100);
}
