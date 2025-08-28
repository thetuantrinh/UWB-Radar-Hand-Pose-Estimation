from keras_flops import get_flops
from tensorflow.keras.models import load_model
import argparse


parser = argparse.ArgumentParser(description="Benchmarking TensorRT models")
parser.add_argument(
    "--saved-model-path",
    default="models/tf/resnet50",
    type=str,
    help="Path where the TensorFlow saved format models are saved"
)

def main():
    args = parser.parse_args()
    model = load_model(args.saved_model_path)
    flops = get_flops(model)

    print("------------MAC and FLOPS------------")
    print(f"got {(flops/2)/(10**6)} M-MAC, flops : {flops/(10**6)} MFLOPS for {args.saved_model_path}")


if __name__ == "__main__":
    main()
