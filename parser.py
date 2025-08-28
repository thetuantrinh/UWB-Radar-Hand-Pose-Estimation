import argparse


def parser():
    parser = argparse.ArgumentParser(
        description="Training handpose estimation with teacher-student architecture"
    )
    parser.add_argument(
        "--checkpoint-path", default="checkpoint/model.pt", type=str
    )
    parser.add_argument(
        "--config-file", default="history/training_information.yaml", type=str
    )
    parser.add_argument("--make-posed-image", action="store_true")
    parser.add_argument("--plot-dot", action="store_true", default=False)
    parser.add_argument(
        "--val-csv",
        default="./val.csv",
        type=str,
        help="should be located relative to the data dir",
    )
    parser.add_argument(
        "--mean",
        default="mean.npy",
        type=str,
        help="should be located relative to the data dir",
    )
    parser.add_argument(
        "--std",
        default="std.npy",
        type=str,
        help="should be located relative to the data dir",
    )
    parser.add_argument("--train-csv", default="./train.csv", type=str)
    parser.add_argument(
        "--data-dir",
        default=".data/",
        type=str,
        help="where the image and radar folders are located",
    )
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--saved-model-path", default="path", type=str)
    parser.add_argument("--background", default="avg_background.npy", type=str)
    parser.add_argument("--arch", default="Residual", type=str)
    parser.add_argument(
        "--criterion",
        default="huber",
        type=str,
        help="available: hubber, mse, mae",
    )
    parser.add_argument("--expansion", default=2, type=int)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--radar-source", default="./paper/demo_radar.npy", type=str
    )
    parser.add_argument(
        "--video-source", default="./paper/demo_video.mp4", type=str
    )
    parser.add_argument("--no-background", action="store_true")
    parser.add_argument(
        "--demo-video-path", default="./paper/demo_posed_video.mp4", type=str
    )

    return parser.parse_args()
