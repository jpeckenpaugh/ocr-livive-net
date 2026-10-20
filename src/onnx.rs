//! Self-contained ONNX v7 / IR v7 protobuf binary serializer.

use crate::centroid::CentroidModel;
use crate::image::IMAGE_LEN;
use crate::knn::KnnModel;
use crate::mlp::MlpModel;
use crate::perceptron::PerceptronModel;

const TENSOR_FLOAT: i32 = 1;

pub struct ProtoWriter {
    buf: Vec<u8>,
}

impl Default for ProtoWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ProtoWriter {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

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
        self.write_tag(field_num, 0);
        self.write_varint(val as u64);
    }

    pub fn write_int32_field(&mut self, field_num: u32, val: i32) {
        self.write_int64_field(field_num, val as i64);
    }

    pub fn write_string_field(&mut self, field_num: u32, s: &str) {
        self.write_tag(field_num, 2);
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

fn make_type_proto_float_tensor(dims: &[i64]) -> ProtoWriter {
    let mut shape_w = ProtoWriter::new();
    for &d in dims {
        let mut dim_w = ProtoWriter::new();
        dim_w.write_int64_field(1, d);
        shape_w.write_msg_field(1, dim_w);
    }

    let mut tensor_type_w = ProtoWriter::new();
    tensor_type_w.write_int32_field(1, TENSOR_FLOAT);
    tensor_type_w.write_msg_field(2, shape_w);

    let mut type_w = ProtoWriter::new();
    type_w.write_msg_field(1, tensor_type_w);
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
        tensor.write_int64_field(1, d);
    }
    tensor.write_int32_field(2, TENSOR_FLOAT);
    tensor.write_string_field(3, name);

    let mut raw_bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        raw_bytes.extend_from_slice(&val.to_le_bytes());
    }
    tensor.write_bytes_field(9, &raw_bytes);
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
        node.write_string_field(1, inp);
    }
    for &out in outputs {
        node.write_string_field(2, out);
    }
    node.write_string_field(3, name);
    node.write_string_field(4, op_type);

    for &(attr_name, attr_i) in attributes {
        let mut attr = ProtoWriter::new();
        attr.write_string_field(1, attr_name);
        attr.write_int64_field(3, attr_i);
        attr.write_int32_field(20, 2);
        node.write_msg_field(5, attr);
    }
    node
}

fn build_model_proto(graph_writer: ProtoWriter, producer_name: &str) -> Vec<u8> {
    let mut model = ProtoWriter::new();
    model.write_int64_field(1, 7);
    model.write_string_field(2, producer_name);
    model.write_string_field(3, "0.1.0");
    model.write_string_field(4, "ocr.livive.net");
    model.write_int64_field(5, 1);
    model.write_msg_field(7, graph_writer);

    let mut opset = ProtoWriter::new();
    opset.write_string_field(1, "");
    opset.write_int64_field(2, 13);
    model.write_msg_field(8, opset);

    model.into_bytes()
}

pub fn export_perceptron(model: &PerceptronModel) -> Vec<u8> {
    let mut graph = ProtoWriter::new();
    graph.write_string_field(2, "perceptron_classifier");

    graph.write_msg_field(11, make_value_info("input_pixels", &[1, IMAGE_LEN as i64]));
    graph.write_msg_field(
        12,
        make_value_info("probabilities", &[1, model.num_classes as i64]),
    );

    graph.write_msg_field(
        5,
        make_float_tensor_initializer("scale_255", &[1], &[255.0]),
    );
    graph.write_msg_field(
        5,
        make_float_tensor_initializer(
            "weights",
            &[model.num_classes as i64, IMAGE_LEN as i64],
            &model.weights,
        ),
    );
    graph.write_msg_field(
        5,
        make_float_tensor_initializer("bias", &[model.num_classes as i64], &model.bias),
    );

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

pub fn export_mlp(model: &MlpModel) -> Vec<u8> {
    let mut graph = ProtoWriter::new();
    graph.write_string_field(2, "mlp_classifier");

    graph.write_msg_field(11, make_value_info("input_pixels", &[1, IMAGE_LEN as i64]));
    graph.write_msg_field(
        12,
        make_value_info("probabilities", &[1, model.num_classes as i64]),
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
        make_float_tensor_initializer("w2", &[model.num_classes as i64, hidden], &model.w2),
    );
    graph.write_msg_field(
        5,
        make_float_tensor_initializer("b2", &[model.num_classes as i64], &model.b2),
    );

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

pub fn export_centroid(model: &CentroidModel) -> Vec<u8> {
    let mut graph = ProtoWriter::new();
    graph.write_string_field(2, "centroid_classifier");

    graph.write_msg_field(11, make_value_info("input_pixels", &[1, IMAGE_LEN as i64]));
    graph.write_msg_field(
        12,
        make_value_info("distances", &[1, model.num_classes as i64]),
    );

    graph.write_msg_field(
        5,
        make_float_tensor_initializer(
            "means",
            &[model.num_classes as i64, IMAGE_LEN as i64],
            &model.means,
        ),
    );

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
    graph.write_msg_field(
        1,
        make_node("Sqrt", &["sum_sq"], &["distances"], "sqrt_dist", &[]),
    );

    build_model_proto(graph, "ocr-livive-net-centroid")
}

pub fn export_knn(model: &KnnModel) -> Vec<u8> {
    let n = model.labels.len();
    assert!(
        n <= 10_000,
        "k-NN ONNX export is supported up to 10,000 samples"
    );

    let mut graph = ProtoWriter::new();
    graph.write_string_field(2, "knn_classifier");

    graph.write_msg_field(11, make_value_info("input_pixels", &[1, IMAGE_LEN as i64]));
    graph.write_msg_field(
        12,
        make_value_info("nearest_labels", &[1, crate::knn::K as i64]),
    );

    let float_images: Vec<f32> = model.images.iter().map(|&p| p as f32).collect();
    graph.write_msg_field(
        5,
        make_float_tensor_initializer("train_images", &[n as i64, IMAGE_LEN as i64], &float_images),
    );

    let float_labels: Vec<f32> = model.labels.iter().map(|&l| l as f32).collect();
    graph.write_msg_field(
        5,
        make_float_tensor_initializer("train_labels", &[n as i64], &float_labels),
    );

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
