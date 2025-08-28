import os
import numpy as np
import torch
import pandas as pd
from utils import logger
from torch.utils.data import Dataset
from ..dsp import process_fft


class LoadRadarDataset(Dataset):
    """
    Dataloader for handpose from radar signals. This class return radar frames,
    hand landmarks, handedness (left or right hand), hand presence (whether hand
    is presence in a radar frames) and the link to the associcated image.
    Radar and RGB image must have the same basename
    """

    def __init__(self, root_dir, csv_label_file, transforms=None):
        # The radar contains all gesture folders.
        self.root_dir = root_dir
        self.csv_label_file = csv_label_file
        self.transforms = transforms
        self.complex_radar = True
        self.fft = False
        self.dtype = np.float32
        self.annotations = pd.read_csv(csv_label_file).fillna(0)
        self.filenames = self.annotations["filename"]
        self.x_0_idx = self.annotations.columns.get_loc("landmark_x_0")
        self.y_20_idx = self.annotations.columns.get_loc("landmark_y_20")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        landmarks = self.annotations.iloc[
            idx, self.x_0_idx : self.y_20_idx + 1
        ]
        landmarks = np.array(landmarks)
        landmarks = landmarks.astype(self.dtype)
        landmarks = landmarks.reshape((21, 2))
        landmarks[:, 0] = landmarks[:, 0] * 640
        landmarks[:, 1] = landmarks[:, 1] * 480
        landmarks = np.round(landmarks)
        landmarks = landmarks.reshape(42)
        handedness = self.annotations["handedness"][idx]
        handedness = np.array(
            [1 if handedness == "Right" else 0], dtype=self.dtype
        )
        hand_presence = np.array(
            [int(self.annotations["hand_presence"][idx])], dtype=self.dtype
        )

        radar_full_path = os.path.join(self.root_dir, self.filenames[idx])
        radar_full_path = radar_full_path.replace("camera", "radar").replace(
            ".png", ".npy"
        )

        if not os.path.isfile(radar_full_path):
            logger.error(
                f"File {radar_full_path} doesn't exist in {self.root_dir}"
            )

        radar_frame = np.load(radar_full_path)

        if self.fft:
            radar_frame = process_fft(radar_frame)
        elif self.complex_radar:
            real, img = np.real(radar_frame), np.imag(radar_frame)
            radar_frame = np.concatenate((real, img), axis=0)

        radar_frame = radar_frame.astype(self.dtype)

        if self.transforms:
            (
                radar_frame,
                landmarks,
                hand_presence,
                handedness,
            ) = self.transforms(
                [radar_frame, landmarks, hand_presence, handedness]
            )

        return radar_frame, landmarks, hand_presence, handedness
