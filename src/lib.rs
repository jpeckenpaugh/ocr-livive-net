#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::needless_range_loop,
    clippy::new_without_default,
    clippy::manual_checked_ops
)]
//! Four transparent digit classifiers (centroid, k-NN, perceptron, MLP) on
//! 28x28 MNIST, plus the wasm-bindgen boundary and ONNX serializer.
//!
//! JS API:
//! - `preprocess(rgba, width, height) -> Uint8Array(784)`
//! - `trainCentroid | trainKnn | trainPerceptron | trainMlp(images, labels, seed) -> ModelHandle`
//!   where `images` is `n * 784` bytes and `labels` is `n` bytes.
//! - `ModelHandle.method()`, `.predict(image)`, `.modelData()`, `.toOnnx() -> Uint8Array`.

use serde::Serialize;
use wasm_bindgen::prelude::*;

pub mod centroid;
pub mod image;
pub mod knn;
pub mod mlp;
pub mod onnx;
pub mod perceptron;
pub mod preprocess;
pub mod rng;

use image::{split_images, Image};

#[wasm_bindgen(js_name = preprocess)]
pub fn run_preprocess(rgba: Vec<u8>, width: u32, height: u32) -> Vec<u8> {
    preprocess::preprocess(&rgba, width, height).into_pixels()
}

pub enum Model {
    Centroid(centroid::CentroidModel),
    Knn(knn::KnnModel),
    Perceptron(perceptron::PerceptronModel),
    Mlp(mlp::MlpModel),
}

#[wasm_bindgen]
pub struct ModelHandle {
    inner: Model,
}

fn to_js<T: Serialize>(value: &T) -> JsValue {
    value
        .serialize(&serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true))
        .expect("serialization failed")
}

#[wasm_bindgen]
impl ModelHandle {
    pub fn method(&self) -> String {
        match self.inner {
            Model::Centroid(_) => "centroid",
            Model::Knn(_) => "knn",
            Model::Perceptron(_) => "perceptron",
            Model::Mlp(_) => "mlp",
        }
        .into()
    }

    pub fn predict(&self, image: Vec<u8>) -> JsValue {
        let image = Image::new(image);
        match &self.inner {
            Model::Centroid(m) => to_js(&centroid::predict(m, &image)),
            Model::Knn(m) => to_js(&knn::predict(m, &image)),
            Model::Perceptron(m) => to_js(&perceptron::predict(m, &image)),
            Model::Mlp(m) => to_js(&mlp::predict(m, &image)),
        }
    }

    /// Static model data.
    #[wasm_bindgen(js_name = modelData)]
    pub fn model_data(&self) -> JsValue {
        #[derive(Serialize)]
        struct KnnData {
            k: usize,
            size: usize,
        }
        match &self.inner {
            Model::Centroid(m) => to_js(m),
            Model::Knn(m) => to_js(&KnnData {
                k: knn::K,
                size: m.labels.len(),
            }),
            Model::Perceptron(m) => to_js(m),
            Model::Mlp(m) => to_js(m),
        }
    }

    /// Export model to binary ONNX format.
    #[wasm_bindgen(js_name = toOnnx)]
    pub fn to_onnx(&self) -> Result<Vec<u8>, JsValue> {
        match &self.inner {
            Model::Centroid(m) => Ok(onnx::export_centroid(m)),
            Model::Knn(m) => {
                if m.labels.len() > 10_000 {
                    Err(JsValue::from_str(
                        "k-NN ONNX export is limited to <= 10,000 samples",
                    ))
                } else {
                    Ok(onnx::export_knn(m))
                }
            }
            Model::Perceptron(m) => Ok(onnx::export_perceptron(m)),
            Model::Mlp(m) => Ok(onnx::export_mlp(m)),
        }
    }
}

#[wasm_bindgen(js_name = trainCentroid)]
pub fn train_centroid(images: Vec<u8>, labels: Vec<u8>, _seed: u32) -> ModelHandle {
    let images = split_images(&images);
    ModelHandle {
        inner: Model::Centroid(centroid::train(&images, &labels)),
    }
}

#[wasm_bindgen(js_name = trainKnn)]
pub fn train_knn(images: Vec<u8>, labels: Vec<u8>, _seed: u32) -> ModelHandle {
    let images = split_images(&images);
    ModelHandle {
        inner: Model::Knn(knn::train(&images, &labels)),
    }
}

#[wasm_bindgen(js_name = trainPerceptron)]
pub fn train_perceptron(images: Vec<u8>, labels: Vec<u8>, seed: u32) -> ModelHandle {
    let images = split_images(&images);
    ModelHandle {
        inner: Model::Perceptron(perceptron::train(&images, &labels, seed as u64)),
    }
}

#[wasm_bindgen(js_name = trainMlp)]
pub fn train_mlp(images: Vec<u8>, labels: Vec<u8>, seed: u32) -> ModelHandle {
    let images = split_images(&images);
    ModelHandle {
        inner: Model::Mlp(mlp::train(&images, &labels, seed as u64)),
    }
}
