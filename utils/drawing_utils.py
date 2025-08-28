import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple
import math
import cv2


HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))

HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))

HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))

HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))

HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))

HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

HAND_CONNECTIONS = frozenset().union(
    *[
        HAND_PALM_CONNECTIONS,
        HAND_THUMB_CONNECTIONS,
        HAND_INDEX_FINGER_CONNECTIONS,
        HAND_MIDDLE_FINGER_CONNECTIONS,
        HAND_RING_FINGER_CONNECTIONS,
        HAND_PINKY_FINGER_CONNECTIONS,
    ]
)


def visualize_traning(
    history,
    title="Training history",
    save_figure=False,
    save_path="./history/plots/training_history.png",
    figsize=(10, 10),
    title_fontsize=16,
):
    # history is a dictionary containing information
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    for idx, key in enumerate(list(history)):
        if "loss" in key:
            axs[0, 0].plot(history[key], label=key)
        if "distance" in key:
            axs[0, 1].plot(history[key], label=key)
        if "train" in key:
            axs[1, 0].plot(history[key], label=key)
        if "val" in key:
            axs[1, 1].plot(history[key], label=key)
        # axs[0, 0].legend()
        # axs[0, 1].legend()
        axs.ravel()[idx].legend()
        axs.ravel()[idx].set_xlabel("epochs")
        axs.ravel()[idx].set_ylabel("val")

    axs[0, 0].set_title("Loss")
    axs[0, 1].set_title("distance")
    axs[1, 0].set_title("train loss & distance")
    axs[1, 1].set_title("val loss & distance")

    fig.suptitle(title, fontsize=title_fontsize)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()


def confusion_matrix():
    # do some stuff here
    print("ok")


def visualize_heatmap(
    frames,
    title="radar heatmaps",
    cmap="hot",
    figsize=(10, 10),
    subxlabel="distance (m)",
    subylabel="velocity (v)",
    multiple_bar=False,
    onebar=True,
    set_xyticks=True,
    shrink=0.8,
    save_path="",
):
    for i in range(1, 9):
        ax = plt.subplot(4, 2, i)
        im = ax.imshow(frames[i - 1,], cmap=cmap, interpolation="nearest")
        ax.set_title(str(i))
        # ax.set_xlabel(str())
        # ax.set_ylabel(str(row))
        if set_xyticks:
            ax.set_xticks([]), ax.set_yticks([])

        plt.colorbar(im)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def visualize_heatmaps(
    frames,
    title="radar heatmaps",
    cmap="hot",
    figsize=(10, 10),
    subxlabel="distance (m)",
    subylabel="velocity (v)",
    multiple_bar=False,
    onebar=True,
    set_xyticks=True,
    shrink=0.8,
    save_path="./history/plots/sample_radar_heatmaps.png",
):
    if len(frames.shape) == 3:
        frames = np.expand_dims(frames, axis=0)
    fig, axs = plt.subplots(frames.shape[0], frames.shape[1], figsize=figsize)
    if frames.shape[0] == 1:
        axs = axs.reshape(1, -1)
    for row in range(frames.shape[0]):
        for col in range(frames.shape[1]):
            ax = axs[row, col]
            im = ax.imshow(
                frames[row, col], cmap=cmap, interpolation="nearest"
            )
            ax.set_xlabel(str(col))
            ax.set_ylabel(str(row))
            if set_xyticks:
                ax.set_xticks([]), ax.set_yticks([])
        if multiple_bar:
            fig.colorbar(im, ax=axs[row, :], shrink=shrink)
    if onebar:
        fig.colorbar(im, ax=axs.ravel().tolist())
    fig.suptitle(title)
    fig.supxlabel(subxlabel)
    fig.supylabel(subylabel)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def scatter_landmark(img_file, gt_landmark, pred, img_size=(640, 480)):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.imshow(img)
    plt.scatter(
        gt_landmark[:, 0] * img_size[0], gt_landmark[:, 1] * img_size[1], c="b"
    )
    plt.scatter(pred[:, 0] * img_size[0], pred[:, 1] * img_size[1], c="r")
    plt.show()


def draw_fingers(
    image,
    landmark_list,
    connections=HAND_CONNECTIONS,
    color=(192, 101, 21),
    thickness=2,
):
    """The landmark_list contains normalized x and y coordinates. Ideally it has 21 landmarks.
    idx_to_coordinates: is a list containing (x, y) coordinates.
    landmark_list: a 2D array of shape (21, 2)
    """

    if np.max(image) < 1:
        image = np.floor(image * 255)

    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list):
        x_px, y_px = normalized_to_pixel_coordinates(
            landmark[0], landmark[1], image.shape[1], image.shape[0]  # x  # y
        )
        idx_to_coordinates[idx] = (x_px, y_px)

    num_landmarks = len(landmark_list)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
        # print(f"connection is {connection}")
        start_idx = connection[0]
        end_idx = connection[1]
        if not (
            0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks
        ):
            raise ValueError(
                f"Landmark index is out of range. Invalid connection "
                f"from landmark #{start_idx} to landmark #{end_idx}."
            )
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            cv2.line(
                image,
                idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx],
                color,
                thickness,
            )

    # # Draws landmark points after finishing the connection lines, which is
    # # aesthetically better.
    # if landmark_drawing_spec:
    #     for idx, landmark_px in idx_to_coordinates.items():
    #         drawing_spec = landmark_drawing_spec[idx] if isinstance(
    #             landmark_drawing_spec, Mapping) else landmark_drawing_spec
    #         # White circle border
    #         circle_border_radius = max(drawing_spec.circle_radius + 1,
    #                              int(drawing_spec.circle_radius * 1.2))
    #         cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
    #                    drawing_spec.thickness)
    #         # Fill color into the circle
    #         cv2.circle(image, landmark_px, drawing_spec.circle_radius,
    #                    drawing_spec.color, drawing_spec.thickness)
    if image.dtype != "uint8":
        image = image.astype(np.uint8)
    return image


def normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int]:
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)

    return x_px, y_px


def scatter_keypoints(
    landmark_list,
    image=None,
    normalized_input=True,
    connections=HAND_CONNECTIONS,
    color=(192, 101, 21),
    resolution=(480, 640, 3),
    thickness=2,
):
    """
    The landmark_list contains normalized x and y coordinates. Ideally it has
    21 landmarks. idx_to_coordinates: is a list containing (x, y) coordinates.
    landmark_list: a 2D array of shape (21, 2)
    """

    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
    elif isinstance(image, np.ndarray):
        if np.max(image) <= 1:
            image = np.floor(image * 255)
    elif image is None:
        image = np.zeros(resolution)

    if image.dtype != "uint8":
        image = image.astype(np.uint8)

    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list):
        if not normalized_input:
            coordinates = int(landmark[0]), int(landmark[1])
        else:
            coordinates = normalized_to_pixel_coordinates(
                landmark[0], landmark[1], image.shape[1], image.shape[0]
            )

        if coordinates is not None:
            idx_to_coordinates[idx] = coordinates

    # Draws the connections if the start and end landmarks are both visible.
    for idx in idx_to_coordinates:
        x, y = idx_to_coordinates[idx]
        if ((x > 0) and (y > 0)) and ((x <= 640) and (y <= 480)):
            cv2.circle(
                image, (x, y), radius=2, color=color, thickness=thickness
            )

    return image
