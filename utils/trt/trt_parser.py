import tensorrt as trt
import os
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def GiB(val):
    return int(val * (1 << 30))


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def build_engine_onnx(
        model_file,
        max_workspace_size=1,
        precision="FP16",
        batch_size=32
):

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    config.max_workspace_size = GiB(max_workspace_size)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    if precision == "FP16":
        if builder.platform_has_fast_fp16:
            print("building engine in FP16")
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        else:
            print("This platform does not support FP32")
    else:
        print("building TRT engine in FP32")
    

    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    engine = builder.build_engine(network, config)
    if engine is None:
        print("failed to create TensorRT engine")
        return None
    print("successfully created TensorRt engine")

    model_name = os.path.split(model_file)[1] # get the base name model
    model_name = model_name.replace(".onnx", "")
    return engine, model_name


def save_engine(
        engine,
        saved_folder,
        name,
        precision="FP16"
):
    save_path = os.path.join(saved_folder, name + "_" + precision + ".engine")
    with open(save_path, "wb") as f:
        f.write(engine.serialize())
    print(f"sucessfully saving the {name} engine")


def load_engine(tensorrt_engine_path):
    # Read the engine from the file and deserialize
    with open(tensorrt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def load_normalized_test_case(test_image, pagelocked_buffer, dtype=trt.float16):
    def normalize_image(image):
        image_arr = image.ravel()
        return image_arr

    np.copyto(pagelocked_buffer, normalize_image(test_image))
    return test_image
