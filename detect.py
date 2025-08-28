import torch
from networks.models import radarnet, radarnet_v2
from networks.get_model import get_model
from tqdm import tqdm
import argparse
from data.loader.inference import InferenceLoader
from utils import logger
from utils import drawing_utils
import numpy as np
import cv2
import os
from parser import parser
from val import get_training_history


def load_model(config_file, checkpoint_path, **kwargs):
    (
        data_dir,
        criterion,
        batch_size,
        expansion,
        arch,
        checkpoint_job_id,
    ) = get_training_history(config_file, checkpoint_path)

    mean, std = np.load(data_dir + "mean.npy"), np.load(data_dir + "std.npy")
    logger.color_text(f"mean: {mean}, std: {std} on the training set")
    background = np.load(data_dir + "avg_background.npy").astype(np.float32)

    logger.debug("Start loading dataset ...")
    loader = InferenceLoader(
        background,
        std,
        mean,
    )

    logger.info("loaded dataset sucessfully")
    logger.debug("Start loading model ...")

    model = get_model(arch)

    loss_fn = torch.nn.MSELoss()
    checkpoints = torch.load(checkpoint_path)
    model.load_state_dict(checkpoints["state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loss_fn.to(device)

    return loader, model, device


def main(**kwargs):

    loader, model, device = load_model(**kwargs)

    radar_sequence = np.load(args.radar_source)
    radar_length = len(radar_sequence)

    if os.path.isfile(args.video_source):
        video_cap = cv2.VideoCapture(args.video_source)
        video_length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert video_length == radar_length, (
            "Video and radar sequence doesn't have equal size sequence: video %s radar %s"
            % (video_length, radar_length)
        )

    saved_video_path = os.path.join(
        os.path.dirname(args.radar_source), "radar_infered_video.mp4"
    )
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_buffer = cv2.VideoWriter(saved_video_path, fourcc, fps, (640, 480))
    pred_color = (0, 255, 0)
    print(
        f"len radar sequence and video {radar_sequence.shape}, {video_length}"
    )
    with torch.no_grad():
        for idx, radar_frame in enumerate(
            tqdm(radar_sequence, desc="inferencing")
        ):
            if os.path.isfile(args.video_source):
                image = video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, image = video_cap.read()
            else:
                image = np.zeros([480, 640, 3])

            radar_frame = loader(radar_frame)
            radar_frame = radar_frame.to(device)

            landmarks, hand_presence, handedness = model(radar_frame)
            landmarks = (
                landmarks.reshape(21, 2)
                .to("cpu")
                .detach()
                .numpy()
                .astype(np.int16)
            )
            infered_image = drawing_utils.scatter_keypoints(
                landmarks,
                image,
                color=pred_color,
                normalized_input=False,
            )
            video_buffer.write(infered_image)

    video_buffer.release()


if __name__ == "__main__":
    args = parser()
    main(**args.__dict__)
