import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks, hand_presence, handedness = sample
        # swap color axis because
        # numpy data: B x C x H x W
        # torch image: B x C x H x W

        return [
            torch.from_numpy(image),
            torch.from_numpy(landmarks),
            torch.from_numpy(hand_presence),
            torch.from_numpy(handedness),
        ]


class Normalize(object):
    """Normailize input data"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        frame = sample[0]
        frame = (frame - self.mean) / self.std

        return [frame, *sample[1:]]


class BackgroundRemoval(object):
    """Normailize input data"""

    def __init__(self, background):
        self.background = background

    def __call__(self, sample):
        radar_frame = sample[0]
        background_subtracted = radar_frame - self.background
        return [background_subtracted, *sample[1:]]
