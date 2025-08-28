import os
import cv2
import h5py
import numpy as np
import logging
import argparse
from tqdm import tqdm


def sync(
    radar_path: str,
    video_path: str,
    video_saved_path: str,
    radar_saved_path: str,
    fps=25,
):
    """
    The total number of radar frames should be smaller than that
    of the radar
    radar_path: dataset/person_name/radar/run.hdf5
    camera_path: dataset/person_name/camera/run.hdf5
    des_folder: synced_dataset/
    This des_folder will then create a path:
    synced_dataset/person_name/camera/run/img.png
    """

    radar_dataset = h5py.File(radar_path, "r")
    radar_idx = list(radar_dataset.keys())
    radar_idx = list(radar_idx)
    radar_idx = np.array(radar_idx, np.int32)
    radar_idx = np.sort(radar_idx)
    total_radar_frames = len(radar_idx)

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    resolution = (640, 480)
    video_out = cv2.VideoWriter(video_saved_path, fourcc, fps, resolution)
    radar_sequence = []

    if not cap.isOpened():
        logging.error("Error opening" + video_path)

    camera_total_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        camera_total_frames += 1

    cam_radar_ratio = camera_total_frames / total_radar_frames

    for idx in tqdm(range(total_radar_frames)):
        radar_frame = process(radar_dataset.get(str(idx)))
        real, img = np.real(radar_frame), np.imag(radar_frame)
        radar_frame = np.concatenate((real, img), axis=0)
        radar_frame = radar_frame.astype(np.float32)

        cam_idx = int(cam_radar_ratio * idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cam_idx)
        ret, image = cap.read()
        if ret:
            radar_sequence.append(radar_frame)
            video_out.write(image)
        else:
            logging.error(f"error while reading frame {idx} in {video_path}")

    cap.release()
    video_out.release()
    radar_sequence = np.asarray(radar_sequence)

    np.save(radar_saved_path, radar_sequence)


def process(adcData, Nc=64, Ns=64):
    adcData = np.reshape(adcData, (8, Nc * Ns), order="F")
    adcData = adcData[[0, 1, 2, 3], :] + 1j * adcData[[4, 5, 6, 7], :]
    adcData = np.array(
        [
            np.reshape(adcData[0], (Nc, Ns)),
            np.reshape(adcData[1], (Nc, Ns)),
            np.reshape(adcData[2], (Nc, Ns)),
            np.reshape(adcData[3], (Nc, Ns)),
        ]
    )
    return adcData


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seperate radar and video frames"
    )
    parser.add_argument("--radar-path", default="dataset.mp4", type=str)
    parser.add_argument("--video-path", default="dataset.hdf5", type=str)
    parser.add_argument("--save-folder", default="./temp/demo1", type=str)
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    camera_name = os.path.basename(args.video_path)
    radar_name = os.path.basename(args.radar_path)
    print(radar_name)
    sync(
        args.radar_path,
        args.video_path,
        os.path.join(args.save_folder, camera_name),
        os.path.join(args.save_folder, radar_name),
    )
