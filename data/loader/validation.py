import os
import numpy as np
import torchvision.transforms as transforms
from ..transformations import ToTensor, BackgroundRemoval, Normalize
from .dataloader import LoadRadarDataset


class ValidationLoader(LoadRadarDataset):
    """
    Dataloader for handpose from radar signals. This class return radar frames,
    hand landmarks, handedness (left or right hand), hand presence (whether hand
    is presence in a radar frames) and the link to the associcated image.
    Radar and RGB image must have the same basename
    Returned value: radar_frame, landmarks, hand_presence, handedness, img_file
    """

    def __init__(self, background, std, mean, **kwargs):
        # The radar contains all gesture folders.
        self.std = np.load(std).astype(np.float32) if type(std) == str else std
        self.mean = (
            np.load(mean).astype(np.float32) if type(mean) == str else mean
        )
        self.background = (
            np.load(background).astype(np.float32)
            if type(background) == str
            else background
        )
        data_transforms = transforms.Compose(
            [
                BackgroundRemoval(self.background),
                ToTensor(),
                Normalize(self.mean, self.std),
            ]
        )
        super().__init__(**kwargs, transforms=data_transforms)

    def __getitem__(self, idx):
        results = super().__getitem__(idx)
        img_full_paths = os.path.join(self.root_dir, self.filenames[idx])
        # return  radar_frame, landmarks, hand_presence, handedness, img_full_paths
        return *results, img_full_paths
