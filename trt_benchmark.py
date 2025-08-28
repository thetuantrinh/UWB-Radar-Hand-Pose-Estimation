import argparse
import numpy as np
import time
import os
import torch
import tensorrt as trt
from utils import common
from utils.trt_parser_v2 import load_normalized_test_case
from utils.trt_parser import build_engine_onnx, load_engine, save_engine
import logging
from data import helper
from data.loader.validation import ValidationLoader
from engine.training_engine import AverageMeter


logging.basicConfig(
    level=logging.DEBUG, format=" \%(asctime)s - \%(levelname)s - \%(message)s"
)

logging.disable(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Convert ONNX models to TensorRT")

parser.add_argument("--device", help="cuda or not", default="cuda:0")
parser.add_argument(
    "--batch-size", type=int, help="data batch size", default=1
)
parser.add_argument(
    "--onnx-model-path",
    help="onnx model path",
    default="models/onnx/ResNet.onnx",
)
parser.add_argument(
    "--tensorrt-path",
    help="tensorrt engine path",
    default="models/trt/ResNet.engine",
)
parser.add_argument(
    "--engine-precision",
    help="precision of TensorRT engine",
    choices=["FP32", "FP16"],
    default="FP16",
)
parser.add_argument(
    "--max-workspace-size", help="workspace of engine", default=0.2, type=float
)
parser.add_argument(
    "--save-engine", help="save TRT engine", action="store_true"
)
parser.add_argument(
    "--load-engine", help="load saved TRT engine", action="store_true"
)
parser.add_argument(
    "--build-engine", help="load saved TRT engine", action="store_true"
)
parser.add_argument(
    "--data-dir",
    help="folder where data are stored",
    type=str,
    default="data/solar/",
)


def main():
    args = parser.parse_args()
    print(args)

    if args.engine_precision == "FP16":
        dtype = trt.float16
        print("converting data to float16")
    else:
        dtype = trt.float32

    if args.build_engine:
        engine, model_name = build_engine_onnx(
            model_file=args.onnx_model_path,
            max_workspace_size=args.max_workspace_size,
            precision=args.engine_precision,
            batch_size=args.batch_size,
        )
        save_engine(
            engine,
            args.tensorrt_engine_path,
            model_name,
            args.engine_precision,
        ) if args.save_engine else None

    if args.load_engine:
        engine = load_engine(args.tensorrt_engine_path)

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    mean, std = np.load(args.data_dir + "mean.npy"), np.load(
        args.data_dir + "std.npy"
    )
    background = np.load(args.data_dir + "avg_background.npy").astype(dtype)
    mean, std, background = (
        mean.astype(dtype),
        std.astype(dtype),
        background.astype(dtype),
    )

    validation_data = ValidationLoader(
        background,
        std,
        mean,
        root_dir=args.data_dir,
        csv_label_file=args.data_dir + "val.csv",
    )
    val_loader = helper.create_batch_loader(
        validation_data, batch_size=1, shuffle=True
    )
    print("loaded dataset sucessfully")
    print("Start loading model ...")

    print("Warming the engine up before benchmarking")
    for i in range(100):
        radar_sample = next(iter(val_loader))[0].unsqueeze(0)
        _ = load_normalized_test_case(
            radar_sample, inputs[0].host, dtype=dtype
        )
        _ = common.do_inference_v2(
            context,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )

    validate(val_loader, context, bindings, inputs, outputs, stream)


def validate(
    val_loader, context, bindings, inputs, outputs, stream, batch_size=1
):
    losses = AverageMeter()
    loss_fn = torch.nn.MSELoss()

    for sample in range(val_loader):
        radar_frame, gt_landmarks, presence, handedness = sample
        radar_frame = load_normalized_test_case(
            radar_frame, inputs[0].host
        )
        trt_outputs = common.do_inference(
            context, bindings, inputs, outputs, stream, batch_size=1
        )
        pred_landmarks = trt_outputs[0]
        regression_loss = loss_fn(pred_landmarks, gt_landmarks)
        losses.update(regression_loss)

    return losses.avg


if __name__ == "__main__":
    main()
