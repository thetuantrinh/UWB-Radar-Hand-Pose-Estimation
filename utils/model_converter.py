import os
import argparse
import numpy as np
import torch


def args_parser():
    parser = argparse.ArgumentParser(description="Benchmarking TensorRT models")
    parser.add_argument("--input-model-path",
                        default="models/h5/some_h5_file.hdf5",
                        type=str,
                        help="path where .h5, savedFormat, or torch models are saved. Input-model can etheir be a path to a model or a directory containing many models"
                        )
    parser.add_argument("--saved-dir",
                        default="models/tf/",
                        type=str,
                        help="path where the converted (.onnx or savedFormat) output models are saved. The saved model name will be identical to the input model except for the extention (.trt, .tflite ...)"
                        )
    parser.add_argument("--all-in-a-dir",
                        action='store_true',
                        help="automate converting all models in a folder to expected format and save"
                        )
    parser.add_argument("--to-onnx",
                        action='store_true'
                        )
    parser.add_argument("--to-TF",
                        action='store_true'
                        )
    parser.add_argument("--to-trt",
                        action='store_true'
                        )
    parser.add_argument("--to-TF-lite",
                        action='store_true'
                        )
    parser.add_argument("--batch-size",
                        default=1,
                        type=int,
                        help="input batch to configure for .onnx conversion"
                        )
    parser.add_argument("--inputs-as-nchw",
                        action='store_true',
                        help="inputs-as-nchw"
                        )
    parser.add_argument("--outputs-as-nchw",
                        action='store_true',
                        help="outputs-as-nchw"
                        )
    parser.add_argument("--torch2onxx",
                        action='store_true',
                        help="torch to onnx"
                        )
    return parser


def tf_to_onnx(model_path,
               batch_size=32,
               to_TF=None,
               to_onnx=None,
               saved_dir="",
               save_model=False,
               inputs_as_nchw=None,
               outputs_as_nchw=None
               ):
    # model_path can be either .hdf5 file or TF savedFormat directory
    # saved_dir is where the converted model is saved
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import tf2onnx

    model = load_model(model_path)
    model_ext = os.path.splitext(model_path)[1]
    base_model_name = os.path.basename(model_path)

    # TensorFlow saved format
    if base_model_name == '':
        base_model_name = model_path[0:len(model_path) - 1]  # remove the /
        base_model_name = os.path.basename(base_model_name)
    # check if model is hdf5
    if model_ext == ".hdf5":
        base_model_name = base_model_name.replace('.hdf5', '')

    if to_TF:
        print(f"converting {base_model_name} to TFsavedModel format")
        model.save(os.path.join(saved_dir, base_model_name)) if save_model else None

    if to_onnx:
        print(f"converting {base_model_name} to .onnx format")
        input_shape = model.input.get_shape().as_list()[1:]
        input_shape.insert(0, batch_size)  # expand to expected shape
        dummy_input = np.random.random(input_shape)
        model.predict(dummy_input)

        spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
        inputs_as_nchw = [model.input.name] if inputs_as_nchw else None
        outputs_as_nchw = [model.output.name] if outputs_as_nchw else None

        output_name = base_model_name + ".onnx"
        output_path = os.path.join(saved_dir, output_name)
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=13,
            output_path=output_path,
            inputs_as_nchw=inputs_as_nchw,
            outputs_as_nchw=outputs_as_nchw
        )
        _ = [n.name for n in model_proto.graph.output]
            
        print(f"successfuly converted {base_model_name} to .onnx file")
    

def main():
    args = args_parser().parse_args()
    print("args", args)
    print("------------------------------------------")

    if args.torch2onnx:
        torch.onnx.export(
            model,
            dummy_input,
            "alexnet.onnx",
            verbose=True,
            input_names=input_names,
            output_names=output_names
        )
    else:
        if args.all_in_a_dir:
            model_names = os.listdir(args.input_model_path)
            for name in model_names:
                model_path = os.path.join(args.input_model_path, name)

                tf_to_onnx(
                    model_path=model_path,
                    batch_size=args.batch_size,
                    to_TF=args.to_TF,
                    to_onnx=args.to_onnx,
                    saved_dir=args.saved_dir,
                    save_model=True
                )

        else:
            tf_to_onnx(
                model_path=args.input_model_path,
                batch_size=args.batch_size,
                to_TF=args.to_TF,
                to_onnx=args.to_onnx,
                saved_dir=args.saved_dir,
                save_model=True,
                inputs_as_nchw=args.inputs_as_nchw,
                outputs_as_nchw=args.outputs_as_nchw            
            )


if __name__ == "__main__":
    main()
