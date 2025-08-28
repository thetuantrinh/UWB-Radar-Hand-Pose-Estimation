import argparse
import os
import cv2
from mediapipe_hand import pose_demo_img


def parser():
    parser = argparse.ArgumentParser(description="MediaPipe parser")
    parser.add_argument(
        "--source",
        help="""
        Input source\n:
        img.jpg    # path to image\n,
        path/      # path to directory\n,
        0          # webcam
        """,
        type=str,
        default="0",
    )
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--save-dir", default="history/handpose/test_img")
    return parser


def main():
    import imghdr

    args = parser().parse_args()
    if args.save_output:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.source == "0":
        cap = cv2.VideoCapture(0)
        pose_demo_img(cap)
        cap.release()
    else:
        image_types = {"jpeg", "png", "gif", "bmp", "ppm", "pgm", "tiff", "webp"}
        if os.path.isdir(args.source):
            img_list = os.listdir(args.source)
            for idx, img_file in enumerate(img_list):
                img_list[idx] = os.path.join(args.source, img_file)
            pose_demo_img(img_list)
        else:
            file_type = imghdr.what(args.source)
            if file_type in image_types:
                print("image input type supported")
                print(f"path {args.source}")
                pose_demo_img([args.source])


if __name__ == "__main__":
    main()
