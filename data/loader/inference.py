import os
import numpy as np
from utils import logger
import torchvision.transforms as transforms
import numpy
from ..transformations import ToTensor, BackgroundRemoval, Normalize
from ..dsp import process_fft


class InferenceLoader:
    """
    Dataloader for handpose from radar signals. This class return radar frames,
    hand landmarks, handedness (left or right hand), hand presence (whether hand
    is presence in a radar frames) and the link to the associcated image.
    Radar and RGB image must have the same basename
    """

    def __init__(self, background, std, mean):
        # The radar contains all gesture folders.
        self.complex_radar = True
        self.std = std
        self.mean = mean
        self.complex_radar = True
        self.fft = False
        self.background = (
            np.load(background).astype(np.float32)
            if type(background) == str
            else background
        )
        self.transforms_ori = transforms.Compose(
            [
                BackgroundRemoval(self.background),
                ToTensor(),
            ]
        )
        self.transforms_full = transforms.Compose(
            [
                BackgroundRemoval(self.background),
                ToTensor(),
                Normalize(self.mean, self.std),                
            ]
        )

        self.transforms = self.transforms_full

    def __call__(self, radar_data):
        if isinstance(radar_data, str):
            if not os.path.isfile(radar_data):
                logger.error(f"File {radar_data} doesn't exist")
            radar_data = np.load(radar_data)
            if self.fft and radar_data.shape[0] == 4:
                radar_data = process_fft(radar_data)
            elif self.complex_radar:
                real, img = np.real(radar_data), np.imag(radar_data)
                radar_data = np.concatenate((real, img), axis=0)

        elif isinstance(radar_data, numpy.ndarray):
            if radar_data.shape[0] == 4:
                real, img = np.real(radar_data), np.imag(radar_data)
                radar_data = np.concatenate((real, img), axis=0)

        radar_data = radar_data.astype(np.float32)
        landmarks = np.zeros(5)
        if self.transforms:
            radar_data, _, _, _ = self.transforms(
                [radar_data, landmarks, np.array([1, 2]), np.array([2, 3])]
            )

        # val_radar = val_radar.unsqueeze(0)  # if torch tensor
        return radar_data.unsqueeze(0)
