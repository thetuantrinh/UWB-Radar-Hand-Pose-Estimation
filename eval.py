import torch
from tqdm import tqdm
from data import helper
from data.loader.validation import ValidationLoader
from utils import logger, drawing_utils, metrics
import numpy as np
import random
from networks.teachers.mediapipe_hand import hand_pose_video, hand_pose_img
from networks.get_model import get_model
import topdown_demo_with_mmdet as topdown_mmpose
import matplotlib.pyplot as plt
import cv2
from time import perf_counter
import os
import yaml
import logging
from parser import parser
from engine.training_engine import AverageMeter

logging.basicConfig(
    level=logging.DEBUG, format=" \%(asctime)s - \%(levelname)s - \%(message)s"
)

logging.disable(level=logging.DEBUG)

torch.manual_seed(42)
np.random.seed(42)


def get_pose_data(img_path):
    '''
    This function is used to pass an RGB image, change brightness with gamma
    correction and then pass to mediapipe to extract pose from MediaPipe data
    structure.
    '''
    def adjust_gamma(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        return cv2.LUT(image, table)

    lm = []
    image = cv2.imread(img_path)
    image = adjust_gamma(image, gamma=random.uniform(0.17, 0.23))
    results = hand_pose_img(image)[0]
    if results.multi_hand_landmarks is not None:
        landmarks = results.multi_hand_landmarks
        for finger_idx, landmark in enumerate(landmarks):
            for idx, keypoint in enumerate(landmark.landmark):
                x = int(keypoint.x * 640)
                y = int(keypoint.y * 480)
                lm.append((x, y))
    else:
        for idx in range(21):
            lm.append((0, 0))

    lm = np.array(lm)
    return lm


def get_training_history(config_file, checkpoint_path):
    with open(config_file, "r") as yaml_file:
        training_history = yaml.load(yaml_file, Loader=yaml.FullLoader)
        logger.debug("load YAML file successfully")

    checkpoint_config_list = os.path.basename(checkpoint_path).split("_")
    arch = checkpoint_config_list[0]
    checkpoint_job_id = checkpoint_config_list[6]
    checkpoint_batch_size = int(checkpoint_config_list[5])
    checkpoint_loss_fn = checkpoint_config_list[3]

    logger.debug("Information extracted from checkpoint path is")
    logger.debug(
        "arch %s job_id %s batch_size %s loss_fn %s"
        % (arch, checkpoint_job_id, checkpoint_batch_size, checkpoint_loss_fn)
    )

    for job_id in training_history:
        if checkpoint_job_id in job_id:
            id_history = training_history[job_id]
            if (
                (arch == id_history["arch"])
                and (checkpoint_batch_size == id_history["batch_size"])
                and (checkpoint_loss_fn == id_history["criterion"])
            ):
                saved_model_path = id_history["saved_model_path"]
                data_dir = id_history["data_dir"]
                criterion = id_history["criterion"]
                batch_size = id_history["batch_size"]
                expansion = id_history["expansion"]
                assert os.path.basename(saved_model_path) in os.path.basename(
                    checkpoint_path
                ), "checkpoint path from input argument doesn't match with yaml training history"
                return (
                    data_dir,
                    criterion,
                    batch_size,
                    expansion,
                    arch,
                    checkpoint_job_id,
                )


def plot_dot(
    val_loader,
    device,
    loss_fn,
    model,
    data_dir,
    checkpoint_job_id,
    batch_size,
    arch,
):
    gt_color = (255, 0, 0)
    pred_color = (0, 0, 255)
    with torch.no_grad():
        for sample in tqdm(val_loader, desc="plotting"):
            (
                radar,
                gt_landmarks,
                gt_hand_presence,
                gt_handedness,
                img_files,
            ) = sample
            radar = radar.to(device)
            gt_landmarks = gt_landmarks.to(device)
            pred_landmarks, pred_hand_presence, pred_handedness = model(radar)

            pred_landmarks = (
                pred_landmarks.reshape(pred_landmarks.shape[0], 21, 2)
                .to("cpu")
                .detach()
                .numpy()
            )
            gt_landmarks = (
                gt_landmarks.reshape(gt_landmarks.shape[0], 21, 2)
                .to("cpu")
                .detach()
                .numpy()
            )
            pred_landmarks = np.round(pred_landmarks).astype(np.int16)
            gt_landmarks = np.round(gt_landmarks).astype(np.int16)

            for idx in range(len(pred_landmarks)):
                image = drawing_utils.scatter_keypoints(
                    gt_landmarks[idx],
                    img_files[idx],
                    color=gt_color,
                    normalized_input=False,
                    thickness=8,
                )
                image = drawing_utils.scatter_keypoints(
                    pred_landmarks[idx],
                    image,
                    color=pred_color,
                    normalized_input=False,
                    thickness=8,
                )

                val_img_path = img_files[idx].replace(data_dir, "")
                val_img_path = val_img_path.replace("camera/", "")
                plt.title(val_img_path)

                img_name = os.path.basename(val_img_path)
                val_img_dir = os.path.dirname(val_img_path)

                save_folder = "./history/inference/JOBID_{}_{}_{}/{}".format(
                    checkpoint_job_id,
                    arch,
                    batch_size,
                    val_img_dir,
                )
                os.makedirs(save_folder, exist_ok=True)

                cv2.imwrite(os.path.join(save_folder, img_name), image)


def main():
    logging.debug("Start of program")
    args = parser()
    (
        data_dir,
        criterion,
        batch_size,
        expansion,
        arch,
        checkpoint_job_id,
    ) = get_training_history(args.config_file, args.checkpoint_path)

    # change the data dir
    data_dir = "./dataset/HandMapping_combined_modified/"

    mean, std = np.load(data_dir + "mean.npy"), np.load(data_dir + "std.npy")
    logger.color_text(f"mean: {mean}, std: {std} on the training set")
    background = np.load(data_dir + "avg_background.npy").astype(np.float32)

    logger.debug("Start loading dataset ...")
    validation_data = ValidationLoader(
        background,
        std,
        mean,
        root_dir=data_dir,
        csv_label_file=data_dir + "val.csv",
    )
    val_loader = helper.create_batch_loader(
        validation_data, batch_size=batch_size, shuffle=True
    )
    logger.info("loaded dataset sucessfully")
    logger.debug("Start loading model ...")

    model = get_model(arch)

    if criterion == "mse":
        loss_fn = torch.nn.MSELoss()

    logger.info("loaded model successfully ...")

    logger.debug(f"start loading trained weights into {arch}")
    checkpoints = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoints["state_dict"])
    logger.info("loaded trained weights successfully")

    logger.info(f"start evaluating {arch} on {data_dir}")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = "cpu"
    model.to(device).eval()
    loss_fn.to(device)

    losses = AverageMeter()

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(val_loader, desc="Evaluating on val set")):
            radar, gt_landmarks, hand_presence, handedness, img_files = sample
            radar = radar.to(device)
            gt_landmarks = gt_landmarks.to(device)
            landmarks, hand_presence, handedness = model(radar)
            regression_loss = loss_fn(landmarks, gt_landmarks)
            losses.update(regression_loss, len(sample))

    if args.plot_dot:
        plot_dot(
            val_loader,
            device,
            loss_fn,
            model,
            data_dir,
            checkpoint_job_id,
            batch_size,
            arch,
        )

    test_radar = next(iter(val_loader))[0][1].unsqueeze(0)
    model.to("cpu")

    for i in range(200):
        _, _, _ = model(test_radar)

    n_tests = 10
    st_time = perf_counter()
    for i in range(n_tests):
        _, _, _ = model(test_radar)
    et_time = perf_counter()

    radar_elapsed_time = 1000 * (et_time - st_time) / n_tests

    image = np.random.rand(480, 640, 3)
    image = image.astype(np.uint8)
    for i in range(n_tests):
        _ = hand_pose_video(image)

    st_time = perf_counter()
    for i in range(n_tests):
        _ = hand_pose_video(image)
    et_time = perf_counter()

    camera_elapsed_time = 1000 * (et_time - st_time) / n_tests
    n_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ) / (10**6)

    print(
        "radar inf_time %.3f camera inf_time %.3f total_params %.3f loss %.3f"
        % (
            radar_elapsed_time,
            camera_elapsed_time,
            n_params,
            losses.avg.to("cpu"),
        )
    )
    radarpose_vs_media_eval_util = metrics.EvalUtil()
    radarpose_vs_mm_eval_util = metrics.EvalUtil()
    media_vs_mm_eval_util = metrics.EvalUtil()
    radarpose_vs_dark_media_eval_util = metrics.EvalUtil()

    model.to("cpu")
    test_data = ValidationLoader(
        background,
        std,
        mean,
        root_dir=data_dir,
        csv_label_file=data_dir + "test_extracted.csv",
    )
    test_loader = helper.create_batch_loader(
        test_data, batch_size=batch_size, shuffle=True
    )

    # get pose generated from mmpose as the groundtruth
    mmpose_args = topdown_mmpose.args
    detector = topdown_mmpose.detector
    pose_estimator = topdown_mmpose.pose_estimator
    visualizer = topdown_mmpose.visualizer

    for sample in tqdm(test_loader, desc="Evaluating perf on different metrics"):
        mmpose_keypoints = []
        mediapipe_dark_keypoints = []
        img_files = sample[4]
        for img_file in img_files:
            mmpose_kp = topdown_mmpose.process_one_image(
                mmpose_args, img_file, detector, pose_estimator, visualizer
            )
            mmpose_keypoints.append(mmpose_kp)
            mediapipe_kp = get_pose_data(img_file)
            mediapipe_dark_keypoints.append(mediapipe_kp)

        mmpose_keypoints = np.array(mmpose_keypoints)
        mediapipe_dark_keypoints = np.array(mediapipe_dark_keypoints)

        mediapie_keypoints = sample[1]
        mediapie_keypoints = (
            mediapie_keypoints.reshape(mediapie_keypoints.shape[0], 21, 2)
            .to("cpu")
            .detach()
            .numpy()
        )

        pred_keypoints, pred_presence, pred_handedness = model(sample[0])
        pred_keypoints = (
            pred_keypoints.reshape(pred_keypoints.shape[0], 21, 2)
            .to("cpu")
            .detach()
            .numpy()
        )
        for idx in range(len(sample[0])):
            # mmpose is used as the generator
            mmpose_keypoint = mmpose_keypoints[idx]
            mmpose_keypoint = mmpose_keypoint.T
            mediapie_keypoint = mediapie_keypoints[idx]
            mediapie_keypoint = mediapie_keypoint.T
            pred_keypoint = pred_keypoints[idx]
            pred_keypoint = pred_keypoint.T
            mediapipe_dark_keypoint = mediapipe_dark_keypoints[idx].T

            # radarpose vs mmpose
            mmpose_vs_keypoint_x = np.logical_and(
                0 < mmpose_keypoint[0], mmpose_keypoint[0] < 640
            )
            mmpose_vs_keypoint_y = np.logical_and(
                0 < mmpose_keypoint[1], mmpose_keypoint[1] < 480
            )
            mmpose_vs_keypoint = np.logical_and(
                mmpose_vs_keypoint_x, mmpose_vs_keypoint_y
            )

            # radarpose vs mediapipe
            media_vs_keypoint_x = np.logical_and(
                0 < mediapie_keypoint[0], mediapie_keypoint[0] < 640
            )
            media_vs_keypoint_y = np.logical_and(
                0 < mediapie_keypoint[1], mediapie_keypoint[1] < 480
            )
            media_vs_keypoint = np.logical_and(media_vs_keypoint_x, media_vs_keypoint_y)

            # mediapipe vs mmpose

            radarpose_vs_mm_eval_util.feed(
                mmpose_keypoint, mmpose_vs_keypoint, pred_keypoint
            )
            radarpose_vs_media_eval_util.feed(
                mediapie_keypoint, media_vs_keypoint, pred_keypoint
            )
            media_vs_mm_eval_util.feed(
                mmpose_keypoint, mmpose_vs_keypoint, mediapie_keypoint
            )
            radarpose_vs_dark_media_eval_util.feed(
                mediapie_keypoint, media_vs_keypoint, mediapipe_dark_keypoint
            )

    (
        radar_vs_mm_mean,
        radar_vs_mm_median,
        radar_vs_mm_auc,
        _,
        _,
    ) = radarpose_vs_mm_eval_util.get_measures(0.0, 40.0, 20)

    (
        radar_vs_media_mean,
        radar_vs_media_median,
        radar_vs_media_auc,
        _,
        _,
    ) = radarpose_vs_media_eval_util.get_measures(0.0, 40.0, 20)

    (
        media_vs_mmpose_mean,
        media_vs_mmpose_median,
        media_vs_mmpose_auc,
        _,
        _,
    ) = media_vs_mm_eval_util.get_measures(0.0, 40.0, 20)

    (
        dark_media_mean,
        dark_media_median,
        dark_media_auc,
        _,
        _,
    ) = radarpose_vs_dark_media_eval_util.get_measures(0.0, 40.0, 20)

    print("Evaluation results on radarpose vs mmpose:")
    print("Average mean EPE: %.3f pixels" % radar_vs_mm_mean)
    print("Average median EPE: %.3f pixels" % radar_vs_mm_median)
    print("Area under curve: %.3f" % radar_vs_mm_auc)

    print("------------------------------------")

    print("Evaluation results on test_data on radar:")
    print("Average mean EPE: %.3f pixels" % radar_vs_media_mean)
    print("Average median EPE: %.3f pixels" % radar_vs_media_median)
    print("Area under curve: %.3f" % radar_vs_media_auc)

    print("------------------------------------")

    print("Evaluation results on mediapipe vs mmpose:")
    print("Average mean EPE: %.3f pixels" % media_vs_mmpose_mean)
    print("Average median EPE: %.3f pixels" % media_vs_mmpose_median)
    print("Area under curve: %.3f" % media_vs_mmpose_auc)

    print("------------------------------------")

    print("Evaluation results on mediapipe vs mmpose on dark cases:")
    print("Average mean EPE: %.3f pixels" % media_vs_mmpose_mean)
    print("Average median EPE: %.3f pixels" % media_vs_mmpose_median)
    print("Area under curve: %.3f" % media_vs_mmpose_auc)

    print("------------------------------------")

    print("Evaluation results on mediapipe vs mmpose on dark cases:")
    print("Average mean EPE: %.3f pixels" % dark_media_mean)
    print("Average median EPE: %.3f pixels" % dark_media_median)
    print("Area under curve: %.3f" % dark_media_auc)


if __name__ == "__main__":
    main()
