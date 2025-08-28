import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks.get_model import get_model
from data import helper, transformations
from data.loader import dataloader
from engine.training_engine import Trainer
from utils import logger, general
import torchvision.transforms as transforms
import os
import wandb
from parser import parser

torch.manual_seed(42)
np.random.seed(42)


def main():
    args = parser()
    print(args)
    dataset = "".join(
        [word for word in args.data_dir.split("/") if "Hand" in word]
    )
    display_name = "%s %s %s %s batch_size %s" % (
        args.arch,
        args.criterion,
        dataset,
        os.environ.get("JOBID"),
        args.batch_size,
    )
    config = {
        "learning_rate": args.lr,
        "architecture": args.arch,
        "dataset": dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "criterion": str(args.criterion),
    }
    wandb.init(
        project="RF-Handpose %s" % (dataset), name=display_name, config=config
    )

    model = get_model(args.arch)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.5, 0.99),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, patience=5, factor=0.5, min_lr=1e-7, verbose=True
    )

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            scheduler = checkpoint["scheduler"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            min_loss = checkpoint["min_loss"]
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    background = np.load(os.path.join(args.data_dir, args.background)).astype(
        np.float32
    )
    mean, std = np.load(os.path.join(args.data_dir, args.mean)), np.load(
        os.path.join(args.data_dir, args.std)
    )
    logger.highlight(
        f"mean: {mean.shape}, std: {std.shape} on the training set"
    )

    composed_transforms = transforms.Compose(
        [
            transformations.BackgroundRemoval(background),
            transformations.ToTensor(),
            transformations.Normalize(mean, std),
        ]
    )

    train_dataset = dataloader.LoadRadarDataset(
        root_dir=args.data_dir,
        csv_label_file=os.path.join(args.data_dir, "train.csv"),
        transforms=composed_transforms,
    )
    val_dataset = dataloader.LoadRadarDataset(
        root_dir=args.data_dir,
        csv_label_file=os.path.join(args.data_dir, "val.csv"),
        transforms=composed_transforms,
    )
    # train_dataset.fft = True
    # val_dataset.fft = True
    train_loader = helper.create_batch_loader(
        train_dataset, batch_size=args.batch_size, num_workers=4
    )
    val_loader = helper.create_batch_loader(
        val_dataset, batch_size=args.batch_size, num_workers=4,
    )

    if args.criterion == "mse":
        criterion = nn.MSELoss()
    elif args.criterion == "hubber":
        criterion = nn.HuberLoss()
    elif args.criterion == "mae":
        criterion = nn.L1Loss()

    bce_criterion1 = nn.BCELoss()
    bce_criterion2 = nn.BCELoss()
    criterions = [criterion, bce_criterion1, bce_criterion2]

    engine = Trainer(
        model,
        train_loader,
        val_loader,
        criterions,
        optimizer,
        scheduler,
        saved_path=args.saved_model_path,
    )
    if args.resume and os.path.isfile(args.resume):
        engine.min_loss = min_loss

    wandb.watch(model)

    engine.train(args.start_epoch, args.epochs)
    wandb.finish()

    JOBID = os.environ.get("JOBID")
    general.export(
        JOBID=JOBID,
        path="./history/training_information.yaml",
        **args.__dict__,
    )


if __name__ == "__main__":
    main()
