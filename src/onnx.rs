//! Self-contained ONNX (Open Neural Network Exchange) v7 / IR v7 protobuf binary serializer.
//!
//! Encodes standard Protobuf wire format (varint, length-delimited) with zero external dependencies.
//! Produces fully compliant `.onnx` models readable by onnxruntime, Netron, tf2onnx, etc.

use crate::centroid::CentroidModel;
use crate::image::{IMAGE_LEN, NUM_CLASSES};
use crate::knn::KnnModel;
use crate::mlp::MlpModel;
use crate::perceptron::PerceptronModel;

// ONNX TensorProto DataType constants:
const TENSOR_FLOAT: i32 = 1;

/// Tiny protobuf binary writer.
pub struct ProtoWriter {
    buf: Vec<u8>,
}

impl ProtoWriter {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Write field header: (field_number << 3) | wire_type.
    pub fn write_tag(&mut self, field_num: u32, wire_type: u8) {
        let tag = (field_num << 3) | (wire_type as u32);
        self.write_varint(tag as u64);
    }

    pub fn write_varint(&mut self, mut val: u64) {
        while val >= 0x80 {
            self.buf.push(((val & 0x7F) as u8) | 0x80);
            val >>= 7;
        }
        self.buf.push((val & 0x7F) as u8);
    }

    pub fn write_int64_field(&mut self, field_num: u32, val: i64) {
        self.write_tag(field_num, 0); // wire type 0 = varint
        self.write_varint(val as u64);
    }

    pub fn write_int32_field(&mut self, field_num: u32, val: i32) {
        self.write_int64_field(field_num, val as i64);
    }

    pub fn write_string_field(&mut self, field_num: u32, s: &str) {
        self.write_tag(field_num, 2); // wire type 2 = length-delimited
        self.write_varint(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }

    pub fn write_bytes_field(&mut self, field_num: u32, bytes: &[u8]) {
        self.write_tag(field_num, 2);
        self.write_varint(bytes.len() as u64);
        self.buf.extend_from_slice(bytes);
    }

    pub fn write_msg_field(&mut self, field_num: u32, sub_writer: ProtoWriter) {
        self.write_bytes_field(field_num, &sub_writer.into_bytes());
    }
}

// ONNX Message Builders

fn make_type_proto_float_tensor(dims: &[i64]) -> ProtoWriter {
    let mut shape_w = ProtoWriter::new();
    for &d in dims {
        let mut dim_w = ProtoWriter::new();
        dim_w.write_int64_field(1, d); // dim_value = 1
        shape_w.write_msg_field(1, dim_w); // repeated Dimension dim = 1;
    }

    let mut tensor_type_w = ProtoWriter::new();
    tensor_type_w.write_int32_field(1, TENSOR_FLOAT); // elem_type = 1 (FLOAT)
    tensor_type_w.write_msg_field(2, shape_w); // shape = 2

    let mut type_w = ProtoWriter::new();
    type_w.write_msg_field(1, tensor_type_w); // tensor_type = 1
    type_w
}

fn make_value_info(name: &str, dims: &[i64]) -> ProtoWriter {
    let mut vi = ProtoWriter::new();
    vi.write_string_field(1, name);
    vi.write_msg_field(2, make_type_proto_float_tensor(dims));
    vi
}

fn make_float_tensor_initializer(name: &str, dims: &[i64], data: &[f32]) -> ProtoWriter {
    let mut tensor = ProtoWriter::new();
    for &d in dims {
        tensor.write_int64_field(1, d); // repeated int64 dims = 1
    }
    tensor.write_int32_field(2, TENSOR_FLOAT); // data_type = 1 (FLOAT)
    tensor.write_string_field(3, name); // name = 3

    // Raw float32 little-endian bytes:
    let mut raw_bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        raw_bytes.extend_from_slice(&val.to_le_bytes());
    }
    tensor.write_bytes_field(9, &raw_bytes); // raw_data = 9

    tensor
}

fn make_node(
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    name: &str,
    attributes: &[(&str, i64)],
) -> ProtoWriter {
    let mut node = ProtoWriter::new();
    for &inp in inputs {
        node.write_string_field(1, inp); // repeated string input = 1
    }
    for &out in outputs {
        node.write_string_field(2, out); // repeated string output = 2
    }
    node.write_string_field(3, name); // name = 3
    node.write_string_field(4, op_type); // op_type = 4

    for &(attr_name, attr_i) in attributes {
        let mut attr = ProtoWriter::new();
        attr.write_string_field(1, attr_name);
        attr.write_int64_field(3, attr_i); // type INT = 2 in attribute_type, field 3 = i
        attr.write_int32_field(20, 2); // type = INT (2)
        node.write_msg_field(5, attr);
    }

    node
}

fn build_model_proto(graph_writer: ProtoWriter, producer_name: &str) -> Vec<u8> {
    let mut model = ProtoWriter::new();
    model.write_int64_field(1, 7); // ir_version = 7
    model.write_string_field(2, producer_name); // producer_name = 2
    model.write_string_field(3, "0.1.0"); // producer_version = 3
    model.write_string_field(4, "ocr.livive.net"); // domain = 4
    model.write_int64_field(5, 1); // model_version = 5
    model.write_msg_field(7, graph_writer); // graph = 7

    // opset_import: domain = "", version = 13
    let mut opset = ProtoWriter::new();
    opset.write_string_field(1, ""); // default domain
    opset.write_int64_field(2, 13); // opset version 13
    model.write_msg_field(8, opset);

    model.into_bytes()
}

// Classifier ONNX Exporters

/// Export Perceptron: Input [1, 784] -> Div(255) -> Gemm (w, b) -> Softmax -> Output [1, 10]
pub fn export_perceptron(model: &PerceptronModel) -> Vec<u8> {
    let mut graph = ProtoWriter::new();
    graph.write_string_field(2, "perceptron_digit_classifier");

    // Inputs & Outputs
    graph.write_msg_field(11, make_value_info("input_pixels", &[1, IMAGE_LEN as i64]));
    graph.write_msg_field(
        12,
        make_value_info("probabilities", &[1, NUM_CLASSES as i64]),
    );

    // Constant 255.0 divisor tensor:
    graph.write_msg_field(
        5,
        make_float_tensor_initializer("scale_255", &[1], &[255.0]),
    );

    // Weights: [784, 10] (transpose of [10, 784] for Gemm without transB, or [10, 784] with transB=1)
    graph.write_msg_field(
        5,
        make_float_tensor_initializer(
            "weights",
            &[NUM_CLASSES as i64, IMAGE_LEN as i64],
            &model.weights,
        ),
    );
    graph.write_msg_field(
        5,
        make_float_tensor_initializer("bias", &[NUM_CLASSES as i64], &model.bias),
    );

    // Node 1: Div (normalize 0..255 to 0..1)
    graph.write_msg_field(
        1,
        make_node(
            "Div",
            &["input_pixels", "scale_255"],
            &["scaled_input"],
            "norm_div",
            &[],
        ),
    );

    // Node 2: Gemm (X * W^T + B)
    graph.write_msg_field(
        1,
        make_node(
            "Gemm",
            &["scaled_input", "weights", "bias"],
            &["logits"],
            "dense_layer",
            &[("transB", 1)],
        ),
    );

    // Node 3: Softmax (axis = 1)
    graph.write_msg_field(
        1,
        make_node(
            "Softmax",
            &["logits"],
            &["probabilities"],
            "prob_softmax",
            &[("axis", 1)],
        ),
    );

    build_model_proto(graph, "ocr-livive-net-perceptron")
}

/// Export MLP: Input [1, 784] -> Div(255) -> Gemm(W1, B1) -> Relu -> Gemm(W2, B2) -> Softmax -> Output [1, 10]
pub fn export_mlp(model: &MlpModel) -> Vec<u8> {
    let mut graph = ProtoWriter::new();
    graph.write_string_field(2, "mlp_digit_classifier");

    graph.write_msg_field(11, make_value_info("input_pixels", &[1, IMAGE_LEN as i64]));
    graph.write_msg_field(
        12,
        make_value_info("probabilities", &[1, NUM_CLASSES as i64]),
    );

    graph.write_msg_field(
        5,
        make_float_tensor_initializer("scale_255", &[1], &[255.0]),
    );

    let hidden = model.hyperparams.hidden_units as i64;
    graph.write_msg_field(
        5,
        make_float_tensor_initializer("w1", &[hidden, IMAGE_LEN as i64], &model.w1),
    );
    graph.write_msg_field(5, make_float_tensor_initializer("b1", &[hidden], &model.b1));
    graph.write_msg_field(
        5,
        make_float_tensor_initializer("w2", &[NUM_CLASSES as i64, hidden], &model.w2),
    );
    graph.write_msg_field(
        5,
        make_float_tensor_initializer("b2", &[NUM_CLASSES as i64], &model.b2),
    );

    // Nodes:
    graph.write_msg_field(
        1,
        make_node(
            "Div",
            &["input_pixels", "scale_255"],
            &["scaled_input"],
            "norm",
            &[],
        ),
    );
    graph.write_msg_field(
        1,
        make_node(
            "Gemm",
            &["scaled_input", "w1", "b1"],
            &["z1"],
            "dense1",
            &[("transB", 1)],
        ),
    );
    graph.write_msg_field(1, make_node("Relu", &["z1"], &["a1"], "relu", &[]));
    graph.write_msg_field(
        1,
        make_node(
            "Gemm",
            &["a1", "w2", "b2"],
            &["logits"],
            "dense2",
            &[("transB", 1)],
        ),
    );
    graph.write_msg_field(
        1,
        make_node(
            "Softmax",
            &["logits"],
            &["probabilities"],
            "softmax",
            &[("axis", 1)],
        ),
    );

    build_model_proto(graph, "ocr-livive-net-mlp")
}

/// Export Centroid: Input [1, 784] -> Distance to 10 class means -> Output [1, 10] Euclidean distances
pub fn export_centroid(model: &CentroidModel) -> Vec<u8> {
    let mut graph = ProtoWriter::new();
    graph.write_string_field(2, "centroid_digit_classifier");

    graph.write_msg_field(11, make_value_info("input_pixels", &[1, IMAGE_LEN as i64]));
    graph.write_msg_field(12, make_value_info("distances", &[1, NUM_CLASSES as i64]));

    // Means [10, 784]
    graph.write_msg_field(
        5,
        make_float_tensor_initializer(
            "means",
            &[NUM_CLASSES as i64, IMAGE_LEN as i64],
            &model.means,
        ),
    );

    // Distances via: ||x - m||_2
    // Node: Sub(input, means) -> [10, 784]
    // Mul(diff, diff) -> Sqr
    // ReduceSum(sqr, axes=[1], keepdims=0) -> [10]
    // Sqrt(sum) -> distances
    graph.write_msg_field(
        1,
        make_node(
            "Sub",
            &["input_pixels", "means"],
            &["diff"],
            "sub_means",
            &[],
        ),
    );
    graph.write_msg_field(
        1,
        make_node("Mul", &["diff", "diff"], &["sq_diff"], "sq_diff", &[]),
    );
    graph.write_msg_field(
        1,
        make_node(
            "ReduceSum",
            &["sq_diff"],
            &["sum_sq"],
            "sum_axes",
            &[("keepdims", 1)],
        ),
    );
    // Unsqueeze/Reshape or Sqrt
    graph.write_msg_field(
        1,
        make_node("Sqrt", &["sum_sq"], &["distances"], "sqrt_dist", &[]),
    );

    build_model_proto(graph, "ocr-livive-net-centroid")
}

/// Export k-NN (up to 10,000 samples)
pub fn export_knn(model: &KnnModel) -> Vec<u8> {
    let n = model.labels.len();
    assert!(
        n <= 10_000,
        "k-NN ONNX export is supported up to 10,000 samples"
    );

    let mut graph = ProtoWriter::new();
    graph.write_string_field(2, "knn_digit_classifier");

    graph.write_msg_field(11, make_value_info("input_pixels", &[1, IMAGE_LEN as i64]));
    graph.write_msg_field(
        12,
        make_value_info("nearest_labels", &[1, crate::knn::K as i64]),
    );

    // Training images: convert u8 to float32 [N, 784]
    let float_images: Vec<f32> = model.images.iter().map(|&p| p as f32).collect();
    graph.write_msg_field(
        5,
        make_float_tensor_initializer("train_images", &[n as i64, IMAGE_LEN as i64], &float_images),
    );

    // Labels as float for tensor export [N]
    let float_labels: Vec<f32> = model.labels.iter().map(|&l| l as f32).collect();
    graph.write_msg_field(
        5,
        make_float_tensor_initializer("train_labels", &[n as i64], &float_labels),
    );

    // Distance calculation: Sub -> Mul -> ReduceSum -> TopK(-dist)
    graph.write_msg_field(
        1,
        make_node(
            "Sub",
            &["input_pixels", "train_images"],
            &["diff"],
            "diff_node",
            &[],
        ),
    );
    graph.write_msg_field(
        1,
        make_node("Mul", &["diff", "diff"], &["sq"], "sq_node", &[]),
    );
    graph.write_msg_field(
        1,
        make_node(
            "ReduceSum",
            &["sq"],
            &["dist_sq"],
            "reduce_node",
            &[("keepdims", 0)],
        ),
    );
    // Negative distance so TopK finds minimum
    let mut neg_one = ProtoWriter::new();
    neg_one.write_int64_field(1, 1);
    neg_one.write_int32_field(2, TENSOR_FLOAT);
    neg_one.write_string_field(3, "neg_one");
    neg_one.write_bytes_field(9, &(-1.0f32).to_le_bytes());
    graph.write_msg_field(5, neg_one);

    graph.write_msg_field(
        1,
        make_node(
            "Mul",
            &["dist_sq", "neg_one"],
            &["neg_dist"],
            "neg_node",
            &[],
        ),
    );

    // TopK with K=3
    graph.write_msg_field(
        1,
        make_node(
            "TopK",
            &["neg_dist"],
            &["top_values", "nearest_labels"],
            "topk_node",
            &[("axis", -1)],
        ),
    );

    build_model_proto(graph, "ocr-livive-net-knn")
}
