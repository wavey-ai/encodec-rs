#!/usr/bin/env python3

"""Split a fixed EnCodec decoder around its ConvTranspose operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def clone(message):
    return type(message).FromString(message.SerializeToString())


def tensor_info(name: str, shape: list[int | str]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    bundle_metadata = json.loads((bundle / "bundle.json").read_text())
    model_path = bundle / bundle_metadata["decode_model"]
    model = onnx.load(model_path)
    nodes = list(model.graph.node)
    initializers = {value.name: value for value in model.graph.initializer}
    original_inputs = {value.name: value for value in model.graph.input}
    producers = {
        name: node for node in nodes for name in node.output if name
    }
    producer_indices = {
        name: index for index, node in enumerate(nodes) for name in node.output if name
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

    conv_indices = [index for index, node in enumerate(nodes) if node.op_type == "ConvTranspose"]
    if len(conv_indices) != 4:
        raise RuntimeError(f"expected four ConvTranspose nodes, found {len(conv_indices)}")

    front_conv_nodes = [
        (index, node)
        for index, node in enumerate(nodes[: conv_indices[0]])
        if node.op_type == "Conv"
    ]
    if len(front_conv_nodes) != 1:
        raise RuntimeError(
            f"expected one decoder-front Conv node, found {len(front_conv_nodes)}"
        )
    front_conv_index, front_conv_node = front_conv_nodes[0]
    front_weight = numpy_helper.to_array(
        initializers[front_conv_node.input[1]]
    ).astype("<f4", copy=False)
    front_bias = numpy_helper.to_array(
        initializers[front_conv_node.input[2]]
    ).astype("<f4", copy=False)
    front_attributes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in front_conv_node.attribute
    }
    front_stride = int(front_attributes["strides"][0])
    front_kernel = int(front_weight.shape[2])
    front_padding = front_kernel - front_stride
    front_padding_right = front_padding // 2
    front_padding_left = front_padding - front_padding_right
    front_mul_node = nodes[front_conv_index + 8]
    front_add_node = nodes[front_conv_index + 9]
    if front_mul_node.op_type != "Mul" or front_add_node.op_type != "Add":
        raise RuntimeError("decoder-front Conv does not have the expected normalization tail")
    front_norm_scale = numpy_helper.to_array(
        initializers[
            resolve_initializer_name(list(front_mul_node.input))
        ]
    ).astype("<f4", copy=False)
    front_norm_bias = numpy_helper.to_array(
        initializers[
            resolve_initializer_name(list(front_add_node.input))
        ]
    ).astype("<f4", copy=False)
    front_weight.tofile(output / "front-conv-weight.f32le")
    front_bias.tofile(output / "front-conv-bias.f32le")
    front_norm_scale.tofile(output / "front-conv-norm-scale.f32le")
    front_norm_bias.tofile(output / "front-conv-norm-bias.f32le")

    lstm_nodes = [node for node in nodes[: conv_indices[0]] if node.op_type == "LSTM"]
    if len(lstm_nodes) != 2:
        raise RuntimeError(f"expected two decoder-front LSTM nodes, found {len(lstm_nodes)}")
    lstm_layers = []
    for layer, node in enumerate(lstm_nodes):
        input_weight = numpy_helper.to_array(initializers[node.input[1]])[0]
        recurrent_weight = numpy_helper.to_array(initializers[node.input[2]])[0]
        raw_bias = numpy_helper.to_array(initializers[node.input[3]])[0]
        hidden_size = int(recurrent_weight.shape[1])
        if input_weight.shape != (4 * hidden_size, hidden_size):
            raise RuntimeError(f"unexpected decoder LSTM input weight shape {input_weight.shape}")
        if recurrent_weight.shape != input_weight.shape:
            raise RuntimeError(
                f"unexpected decoder LSTM recurrent weight shape {recurrent_weight.shape}"
            )
        input_weight.astype("<f4", copy=False).tofile(
            output / f"front-lstm-{layer}-input-weight.f32le"
        )
        recurrent_weight.astype("<f4", copy=False).tofile(
            output / f"front-lstm-{layer}-recurrent-weight.f32le"
        )
        combined_bias = (
            raw_bias[: 4 * hidden_size] + raw_bias[4 * hidden_size :]
        ).astype("<f4", copy=False)
        combined_bias.tofile(output / f"front-lstm-{layer}-bias.f32le")
        lstm_layers.append(
            {
                "layer": layer,
                "name": node.name,
                "inputSize": hidden_size,
                "hiddenSize": hidden_size,
                "gateSize": 4 * hidden_size,
            }
        )

    codebooks = []
    for node in nodes[: conv_indices[0]]:
        if node.op_type != "Gather" or node.input[0] not in initializers:
            continue
        candidate = numpy_helper.to_array(initializers[node.input[0]])
        if candidate.ndim == 2 and candidate.shape[0] == 1024:
            codebooks.append(candidate.astype("<f4", copy=False))
    expected_codebooks = int(bundle_metadata["num_codebooks"])
    if len(codebooks) < expected_codebooks:
        raise RuntimeError(
            f"expected at least {expected_codebooks} decoder codebooks, found {len(codebooks)}"
        )
    codebooks = codebooks[:expected_codebooks]
    embeddings = np.stack(codebooks)
    embeddings.tofile(output / "front-rvq-embeddings.f32le")

    frame_length = int(bundle_metadata["frame_length"])
    layer_metadata = []
    input_time = frame_length
    for layer, node_index in enumerate(conv_indices):
        node = nodes[node_index]
        weight = numpy_helper.to_array(initializers[node.input[1]]).astype("<f4", copy=False)
        bias = numpy_helper.to_array(initializers[node.input[2]]).astype("<f4", copy=False)
        stride = int(next(helper.get_attribute_value(a)[0] for a in node.attribute if a.name == "strides"))
        kernel = int(weight.shape[2])
        if kernel != stride * 2:
            raise RuntimeError(f"layer {layer} does not have kernel = 2 * stride")
        raw_output_time = (input_time - 1) * stride + kernel
        cropped_output_time = input_time * stride
        crop_total = raw_output_time - cropped_output_time
        crop_right = crop_total // 2
        crop_left = crop_total - crop_right
        weight.tofile(output / f"layer-{layer}-weight.f32le")
        bias.tofile(output / f"layer-{layer}-bias.f32le")
        norm_mul_node = nodes[node_index + 8]
        norm_add_node = nodes[node_index + 9]
        if norm_mul_node.op_type != "Mul" or norm_add_node.op_type != "Add":
            raise RuntimeError(
                f"ConvTranspose layer {layer} does not have the expected normalization tail"
            )
        norm_scale = numpy_helper.to_array(
            initializers[resolve_initializer_name(list(norm_mul_node.input))]
        ).astype("<f4", copy=False)
        norm_bias = numpy_helper.to_array(
            initializers[resolve_initializer_name(list(norm_add_node.input))]
        ).astype("<f4", copy=False)
        norm_scale.tofile(output / f"layer-{layer}-norm-scale.f32le")
        norm_bias.tofile(output / f"layer-{layer}-norm-bias.f32le")
        layer_metadata.append(
            {
                "layer": layer,
                "inputChannels": int(weight.shape[0]),
                "outputChannels": int(weight.shape[1]),
                "kernel": kernel,
                "stride": stride,
                "inputTime": input_time,
                "rawOutputTime": raw_output_time,
                "croppedOutputTime": cropped_output_time,
                "cropLeft": crop_left,
                "cropRight": crop_right,
                "inputName": node.input[0],
                "outputName": node.output[0],
            }
        )
        input_time = cropped_output_time

    all_conv_nodes = [
        (index, node) for index, node in enumerate(nodes) if node.op_type == "Conv"
    ]
    if len(all_conv_nodes) != 14:
        raise RuntimeError(f"expected 14 decoder Conv nodes, found {len(all_conv_nodes)}")
    post_conv_nodes = all_conv_nodes[1:]
    post_conv_layers = []
    block_layers = []
    cursor = 0
    for block, decoder_layer in enumerate(layer_metadata):
        block_indices = []
        for role in ("shortcut", "mainReduce", "mainExpand"):
            node_index, node = post_conv_nodes[cursor]
            weight = numpy_helper.to_array(initializers[node.input[1]]).astype(
                "<f4", copy=False
            )
            bias = numpy_helper.to_array(initializers[node.input[2]]).astype(
                "<f4", copy=False
            )
            attributes = {
                attribute.name: helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            stride = int(attributes["strides"][0])
            kernel = int(weight.shape[2])
            padding = kernel - stride
            padding_right = padding // 2
            padding_left = padding - padding_right
            norm_mul_node = nodes[node_index + 8]
            norm_add_node = nodes[node_index + 9]
            if norm_mul_node.op_type != "Mul" or norm_add_node.op_type != "Add":
                raise RuntimeError(
                    f"post Conv layer {cursor} does not have the expected normalization tail"
                )
            norm_scale = numpy_helper.to_array(
                initializers[resolve_initializer_name(list(norm_mul_node.input))]
            ).astype("<f4", copy=False)
            norm_bias = numpy_helper.to_array(
                initializers[resolve_initializer_name(list(norm_add_node.input))]
            ).astype("<f4", copy=False)
            weight.tofile(output / f"post-conv-{cursor}-weight.f32le")
            bias.tofile(output / f"post-conv-{cursor}-bias.f32le")
            norm_scale.tofile(output / f"post-conv-{cursor}-norm-scale.f32le")
            norm_bias.tofile(output / f"post-conv-{cursor}-norm-bias.f32le")
            post_conv_layers.append(
                {
                    "layer": cursor,
                    "block": block,
                    "role": role,
                    "name": node.name,
                    "inputChannels": int(weight.shape[1]),
                    "outputChannels": int(weight.shape[0]),
                    "kernel": kernel,
                    "stride": stride,
                    "inputTime": decoder_layer["croppedOutputTime"],
                    "paddingLeft": padding_left,
                    "paddingRight": padding_right,
                    "paddedInputTime": decoder_layer["croppedOutputTime"] + padding,
                    "outputTime": decoder_layer["croppedOutputTime"],
                }
            )
            block_indices.append(cursor)
            cursor += 1
        block_layers.append(block_indices)

    if cursor != len(post_conv_nodes) - 1:
        raise RuntimeError("decoder post-Conv block accounting is inconsistent")
    final_node_index, final_node = post_conv_nodes[cursor]
    final_weight = numpy_helper.to_array(initializers[final_node.input[1]]).astype(
        "<f4", copy=False
    )
    final_bias = numpy_helper.to_array(initializers[final_node.input[2]]).astype(
        "<f4", copy=False
    )
    final_attributes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in final_node.attribute
    }
    final_stride = int(final_attributes["strides"][0])
    final_kernel = int(final_weight.shape[2])
    final_padding = final_kernel - final_stride
    final_padding_right = final_padding // 2
    final_padding_left = final_padding - final_padding_right
    final_mul_node = nodes[final_node_index + 8]
    final_add_node = nodes[final_node_index + 9]
    if final_mul_node.op_type != "Mul" or final_add_node.op_type != "Add":
        raise RuntimeError("final decoder Conv does not have the expected normalization tail")
    final_norm_scale = numpy_helper.to_array(
        initializers[resolve_initializer_name(list(final_mul_node.input))]
    ).astype("<f4", copy=False)
    final_norm_bias = numpy_helper.to_array(
        initializers[resolve_initializer_name(list(final_add_node.input))]
    ).astype("<f4", copy=False)
    final_weight.tofile(output / "final-conv-weight.f32le")
    final_bias.tofile(output / "final-conv-bias.f32le")
    final_norm_scale.tofile(output / "final-conv-norm-scale.f32le")
    final_norm_bias.tofile(output / "final-conv-norm-bias.f32le")
    final_conv = {
        "inputChannels": int(final_weight.shape[1]),
        "outputChannels": int(final_weight.shape[0]),
        "kernelOutputChannels": 8,
        "kernel": final_kernel,
        "stride": final_stride,
        "inputTime": int(bundle_metadata["segment_samples"]),
        "paddingLeft": final_padding_left,
        "paddingRight": final_padding_right,
        "paddedInputTime": int(bundle_metadata["segment_samples"]) + final_padding,
        "outputTime": int(bundle_metadata["segment_samples"]),
    }

    ranges = []
    start = 0
    for node_index in conv_indices:
        ranges.append((start, node_index))
        start = node_index + 1
    ranges.append((start, len(nodes)))

    for stage, (start, end) in enumerate(ranges):
        if stage == 0:
            stage_inputs = [clone(value) for value in model.graph.input]
        else:
            previous = layer_metadata[stage - 1]
            stage_inputs = [
                tensor_info(
                    previous["outputName"],
                    ["batch", previous["outputChannels"], previous["rawOutputTime"]],
                )
            ]

        selected_indices = set(range(start, end))
        stage_inputs = close_stage_dependencies(
            stage,
            nodes,
            selected_indices,
            stage_inputs,
            original_inputs,
            initializers,
            producer_indices,
        )
        stage_nodes = [clone(nodes[index]) for index in sorted(selected_indices)]

        if stage < len(layer_metadata):
            layer = layer_metadata[stage]
            stage_outputs = [
                tensor_info(layer["inputName"], ["batch", layer["inputChannels"], layer["inputTime"]])
            ]
        else:
            stage_outputs = [
                tensor_info(
                    model.graph.output[0].name,
                    ["batch", int(bundle_metadata["channels"]), int(bundle_metadata["segment_samples"])],
                )
            ]

        referenced_initializers = {
            name
            for node in stage_nodes
            for name in node.input
            if name in initializers
        }
        stage_initializers = [clone(initializers[name]) for name in sorted(referenced_initializers)]
        validate_inputs(stage, stage_nodes, stage_inputs, stage_initializers)

        graph = helper.make_graph(
            stage_nodes,
            f"encodec_decoder_stage_{stage}",
            stage_inputs,
            stage_outputs,
            stage_initializers,
        )
        stage_model = helper.make_model(
            graph,
            opset_imports=[clone(value) for value in model.opset_import],
            ir_version=model.ir_version,
            producer_name="encodec-rs-custom-decoder",
        )
        onnx.checker.check_model(stage_model)
        onnx.save(stage_model, output / f"stage-{stage}.onnx")

    report = {
        "sourceModel": str(model_path),
        "onnxFree": True,
        "frameLength": frame_length,
        "numCodebooks": int(bundle_metadata["num_codebooks"]),
        "channels": int(bundle_metadata["channels"]),
        "segmentSamples": int(bundle_metadata["segment_samples"]),
        "front": {
            "rvq": {
                "codebooks": int(embeddings.shape[0]),
                "entries": int(embeddings.shape[1]),
                "dimension": int(embeddings.shape[2]),
            },
            "conv": {
                "inputChannels": int(front_weight.shape[1]),
                "outputChannels": int(front_weight.shape[0]),
                "kernel": front_kernel,
                "stride": front_stride,
                "inputTime": frame_length,
                "paddingLeft": front_padding_left,
                "paddingRight": front_padding_right,
                "paddedInputTime": frame_length + front_padding,
                "outputTime": frame_length,
            },
            "lstmLayers": lstm_layers,
        },
        "post": {
            "convLayers": post_conv_layers,
            "blocks": block_layers,
            "finalConv": final_conv,
        },
        "layers": layer_metadata,
        "stages": len(ranges),
    }
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "frameLength": report["frameLength"],
                "numCodebooks": report["numCodebooks"],
                "transposeLayers": len(layer_metadata),
                "onnxFree": report["onnxFree"],
            }
        )
    )


def validate_inputs(stage, nodes, graph_inputs, initializers) -> None:
    available = {value.name for value in graph_inputs}
    available.update(value.name for value in initializers)
    for node in nodes:
        missing = [name for name in node.input if name and name not in available]
        if missing:
            raise RuntimeError(f"stage {stage} node {node.name!r} has missing inputs {missing}")
        available.update(name for name in node.output if name)


def close_stage_dependencies(
    stage,
    nodes,
    selected_indices,
    graph_inputs,
    original_inputs,
    initializers,
    producer_indices,
):
    input_names = {value.name for value in graph_inputs}
    while True:
        produced = {
            name
            for index in selected_indices
            for name in nodes[index].output
            if name
        }
        missing = {
            name
            for index in selected_indices
            for name in nodes[index].input
            if name
            and name not in produced
            and name not in input_names
            and name not in initializers
        }
        if not missing:
            return graph_inputs

        changed = False
        for name in sorted(missing):
            if name in original_inputs:
                graph_inputs.append(clone(original_inputs[name]))
                input_names.add(name)
                changed = True
                continue
            producer_index = producer_indices.get(name)
            if producer_index is None:
                raise RuntimeError(f"stage {stage} has unresolved input {name!r}")
            producer = nodes[producer_index]
            if producer.op_type not in {"Constant", "Identity"}:
                raise RuntimeError(
                    f"stage {stage} would cross a dynamic {producer.op_type} dependency for {name!r}"
                )
            selected_indices.add(producer_index)
            changed = True
        if not changed:
            raise RuntimeError(f"stage {stage} dependency closure made no progress")


if __name__ == "__main__":
    main()
