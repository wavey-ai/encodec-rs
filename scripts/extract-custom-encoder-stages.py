#!/usr/bin/env python3

"""Split a fixed EnCodec encoder around its recurrent and quantizer stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, utils


PRE_LSTM = "/encoder/model/model.13/Transpose_output_0"
POST_LSTM = "/encoder/model/model.13/Transpose_1_output_0"
LATENT = "/encoder/model/model.15/conv/norm/Add_output_0"
FRONT_CUTS = [
    ("initial_conv", "/encoder/model/model.0/conv/norm/Add_output_0"),
    ("residual_32", "/encoder/model/model.1/Add_output_0"),
    ("downsample_64", "/encoder/model/model.3/conv/norm/Add_output_0"),
    ("residual_64", "/encoder/model/model.4/Add_output_0"),
    ("downsample_128", "/encoder/model/model.6/conv/norm/Add_output_0"),
    ("residual_128", "/encoder/model/model.7/Add_output_0"),
    ("downsample_256", "/encoder/model/model.9/conv/norm/Add_output_0"),
    ("residual_256", "/encoder/model/model.10/Add_output_0"),
    ("downsample_512", "/encoder/model/model.12/conv/norm/Add_output_0"),
    ("pre_lstm_transpose", PRE_LSTM),
]


def clone(message):
    return type(message).FromString(message.SerializeToString())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    bundle_metadata = json.loads((bundle / "bundle.json").read_text())
    model_path = bundle / bundle_metadata["encode_model"]
    inferred_path = output / "encode-frame-inferred.onnx"
    inferred = onnx.shape_inference.infer_shapes(onnx.load(model_path))
    onnx.save(inferred, inferred_path)

    initializers = {value.name: value for value in inferred.graph.initializer}
    nodes = list(inferred.graph.node)
    producers = {
        output_name: node
        for node in nodes
        for output_name in node.output
        if output_name
    }

    def resolve_initializer_name(names: list[str]) -> str:
        for name in names:
            current = name
            while current not in initializers:
                producer = producers.get(current)
                if producer is None or producer.op_type != "Identity":
                    break
                current = producer.input[0]
            if current in initializers:
                return current
        raise RuntimeError(f"inputs do not resolve to an initializer: {names}")
    conv_nodes = [
        (index, node) for index, node in enumerate(nodes) if node.op_type == "Conv"
    ]
    if len(conv_nodes) != 18:
        raise RuntimeError(f"expected 18 Conv nodes, found {len(conv_nodes)}")

    lstm_nodes = [node for node in nodes if node.op_type == "LSTM"]
    if len(lstm_nodes) != 2:
        raise RuntimeError(f"expected 2 LSTM nodes, found {len(lstm_nodes)}")
    lstm_layers = []
    for layer, node in enumerate(lstm_nodes):
        input_weight = numpy_helper.to_array(initializers[node.input[1]])[0]
        recurrent_weight = numpy_helper.to_array(initializers[node.input[2]])[0]
        raw_bias = numpy_helper.to_array(initializers[node.input[3]])[0]
        hidden_size = int(recurrent_weight.shape[1])
        if input_weight.shape != (4 * hidden_size, hidden_size):
            raise RuntimeError(f"unexpected LSTM input weight shape {input_weight.shape}")
        if recurrent_weight.shape != input_weight.shape:
            raise RuntimeError(
                f"unexpected LSTM recurrent weight shape {recurrent_weight.shape}"
            )
        input_weight = input_weight.astype("<f4", copy=False)
        recurrent_weight = recurrent_weight.astype("<f4", copy=False)
        combined_bias = (raw_bias[: 4 * hidden_size] + raw_bias[4 * hidden_size :]).astype(
            "<f4", copy=False
        )
        input_weight.tofile(output / f"lstm-{layer}-input-weight.f32le")
        recurrent_weight.tofile(output / f"lstm-{layer}-recurrent-weight.f32le")
        combined_bias.tofile(output / f"lstm-{layer}-bias.f32le")
        lstm_layers.append(
            {
                "layer": layer,
                "name": node.name,
                "inputSize": hidden_size,
                "hiddenSize": hidden_size,
                "gateSize": 4 * hidden_size,
                "combinedInputSize": 2 * hidden_size,
            }
        )

    quantizer_matmuls = [node for node in nodes if node.op_type == "MatMul"]
    if len(quantizer_matmuls) != int(bundle_metadata["num_codebooks"]):
        raise RuntimeError(
            f"expected {bundle_metadata['num_codebooks']} quantizer MatMul nodes, "
            f"found {len(quantizer_matmuls)}"
        )
    codebooks = []
    for node in quantizer_matmuls:
        embedding = numpy_helper.to_array(initializers[node.input[1]]).T.astype(
            "<f4", copy=False
        )
        codebooks.append(embedding)
    embeddings = np.stack(codebooks)
    embedding_norms = np.sum(embeddings * embeddings, axis=2, dtype=np.float32)
    embeddings.tofile(output / "rvq-embeddings.f32le")
    embedding_norms.astype("<f4", copy=False).tofile(output / "rvq-norms.f32le")

    conv_layers = []
    current_time = int(bundle_metadata["segment_samples"])
    for layer, (node_index, node) in enumerate(conv_nodes):
        input_time = current_time
        weight = numpy_helper.to_array(initializers[node.input[1]]).astype("<f4", copy=False)
        bias = numpy_helper.to_array(initializers[node.input[2]]).astype("<f4", copy=False)
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        stride = int(attributes["strides"][0])
        kernel = int(weight.shape[2])
        padding = kernel - stride
        padding_right = padding // 2
        padding_left = padding - padding_right
        output_time = input_time // stride
        current_time = output_time
        weight.tofile(output / f"conv-{layer}-weight.f32le")
        bias.tofile(output / f"conv-{layer}-bias.f32le")

        mul_node = nodes[node_index + 8]
        add_node = nodes[node_index + 9]
        if mul_node.op_type != "Mul" or add_node.op_type != "Add":
            raise RuntimeError(f"Conv layer {layer} does not have the expected normalization tail")
        norm_scale_name = resolve_initializer_name(list(mul_node.input))
        norm_bias_name = resolve_initializer_name(list(add_node.input))
        norm_scale = numpy_helper.to_array(initializers[norm_scale_name]).astype("<f4", copy=False)
        norm_bias = numpy_helper.to_array(initializers[norm_bias_name]).astype("<f4", copy=False)
        norm_scale.tofile(output / f"conv-{layer}-norm-scale.f32le")
        norm_bias.tofile(output / f"conv-{layer}-norm-bias.f32le")

        conv_graph = helper.make_graph(
            [clone(node)],
            f"encodec_encoder_conv_{layer}",
            [
                helper.make_tensor_value_info(
                    node.input[0],
                    TensorProto.FLOAT,
                    [1, int(weight.shape[1]), input_time + padding],
                )
            ],
            [
                helper.make_tensor_value_info(
                    node.output[0],
                    TensorProto.FLOAT,
                    [1, int(weight.shape[0]), output_time],
                )
            ],
            [clone(initializers[node.input[1]]), clone(initializers[node.input[2]])],
        )
        conv_model = helper.make_model(
            conv_graph,
            opset_imports=[clone(value) for value in inferred.opset_import],
            ir_version=inferred.ir_version,
            producer_name="encodec-rs-custom-encoder",
        )
        onnx.checker.check_model(conv_model)
        onnx.save(conv_model, output / f"conv-{layer}.onnx")

        conv_layers.append(
            {
                "layer": layer,
                "name": node.name,
                "inputChannels": int(weight.shape[1]),
                "outputChannels": int(weight.shape[0]),
                "kernel": kernel,
                "stride": stride,
                "inputTime": input_time,
                "paddingLeft": padding_left,
                "paddingRight": padding_right,
                "paddedInputTime": input_time + padding,
                "outputTime": output_time,
            }
        )

    stages = [
        {
            "name": "convolutional_front",
            "inputs": ["audio"],
            "outputs": [PRE_LSTM, "scale"],
        },
        {
            "name": "recurrent",
            "inputs": [PRE_LSTM],
            "outputs": [POST_LSTM],
        },
        {
            "name": "latent_projection",
            "inputs": [POST_LSTM],
            "outputs": [LATENT],
        },
        {
            "name": "residual_vector_quantizer",
            "inputs": [LATENT],
            "outputs": ["codes"],
        },
    ]

    for index, stage in enumerate(stages):
        stage_path = output / f"stage-{index}.onnx"
        utils.extract_model(
            inferred_path,
            stage_path,
            stage["inputs"],
            stage["outputs"],
            check_model=True,
        )

    front_stages = []
    previous = "audio"
    for index, (name, output_name) in enumerate(FRONT_CUTS):
        stage_outputs = [output_name]
        if index == 0:
            stage_outputs.append("scale")
        stage = {
            "name": name,
            "inputs": [previous],
            "outputs": stage_outputs,
        }
        utils.extract_model(
            inferred_path,
            output / f"front-stage-{index}.onnx",
            stage["inputs"],
            stage["outputs"],
            check_model=True,
        )
        front_stages.append(stage)
        previous = output_name

    inferred_path.unlink()
    report = {
        "sourceModel": str(model_path),
        "sampleRate": int(bundle_metadata["sample_rate"]),
        "channels": int(bundle_metadata["channels"]),
        "segmentSamples": int(bundle_metadata["segment_samples"]),
        "segmentStride": int(bundle_metadata["segment_stride"]),
        "frameLength": int(bundle_metadata["frame_length"]),
        "numCodebooks": int(bundle_metadata["num_codebooks"]),
        "stages": stages,
        "frontStages": front_stages,
        "convLayers": conv_layers,
        "lstmLayers": lstm_layers,
        "rvq": {
            "codebooks": int(embeddings.shape[0]),
            "entries": int(embeddings.shape[1]),
            "dimension": int(embeddings.shape[2]),
        },
    }
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "frameLength": report["frameLength"],
                "numCodebooks": report["numCodebooks"],
                "convLayers": len(conv_layers),
                "lstmLayers": len(lstm_layers),
            }
        )
    )


if __name__ == "__main__":
    main()
