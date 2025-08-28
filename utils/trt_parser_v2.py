import numpy as np

try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

import tensorrt as trt

from UTILS import common

DTYPE = trt.float16


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine_onnx(
    model_file, max_workspace_size=0.2, precision="FP16", batch_size=1
):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = common.GiB(0.5)
    if precision == "FP16":
        if builder.platform_has_fast_fp16:
            print("converting to FP16")
            config.set_flag(trt.BuilderFlag.FP16)

    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print(f"the engine has been built")
    return builder.build_engine(network, config)


def load_normalized_test_case(
    test_image, pagelocked_buffer, dtype=trt.float16
):
    def normalize_image(image):
        image_arr = image.ravel()
        return image_arr

    np.copyto(pagelocked_buffer, normalize_image(test_image))
    return test_image
