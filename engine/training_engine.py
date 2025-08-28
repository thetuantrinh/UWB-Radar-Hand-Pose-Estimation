from utils import logger
import torch
import os
import wandb
from shutil import copyfile


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        saved_path=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.criterion = criterion
        self.model.to(self.device)
        self.saved_path = saved_path
        self.regression_train_loss = []
        self.regression_val_loss = []
        self.min_regression_loss = torch.inf
        self.hand_presence_train_loss = []
        self.hand_presence_val_loss = []
        self.min_presence_loss = torch.inf
        self.handedness_train_loss = []
        self.handedness_val_loss = []
        self.min_handedness_loss = torch.inf
        self.history = []
        self.print_freq = 10
        self.best_checkpoint_path = ""
        torch.autograd.set_detect_anomaly(True)
        wandb.watch(self.model)
        self.epochs = 0
        wandb.log(
            {
                "Total parameters": sum(
                    [p.numel() for p in model.parameters()]
                ),
                "Trainable parameters": sum(
                    [p.numel() for p in model.parameters() if p.requires_grad]
                ),
            }
        )

    def train(self, start_epoch, epochs):
        logger.info(
            f"Training model on {self.device} for {epochs} epochs epoch {start_epoch}"
        )
        self.epochs = epochs
        for epoch in range(start_epoch, epochs):
            self.train_epoch(epoch)
            self.val_epoch(epoch)
            wandb.log(
                {
                    "epoch": epoch,
                    "regression_train_loss": self.regression_train_loss[-1],
                    "regression_val_loss": self.regression_val_loss[-1],
                    "hand_presence_train_loss": self.hand_presence_train_loss[
                        -1
                    ],
                    "hand_presence_val_loss": self.hand_presence_val_loss[-1],
                    "handedness_train_loss": self.handedness_train_loss[-1],
                    "handedness_val_loss": self.handedness_val_loss[-1],
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )
            self.history = {
                "train_loss": {
                    "regression_loss": self.regression_val_loss,
                    "handedness_loss": self.handedness_train_loss,
                    "hand_presence_loss": self.hand_presence_train_loss,
                },
                "val_loss": {
                    "regresion_loss": self.regression_val_loss,
                    "handedness_loss": self.handedness_val_loss,
                    "hand_presence_loss": self.hand_presence_val_loss,
                },
                "min_loss": self.min_regression_loss,
            }

    def train_epoch(self, epoch):
        self.model.train()
        regression_losses = AverageMeter()
        handedness_losses = AverageMeter()
        hand_presence_losses = AverageMeter()
        for i, data_batch in enumerate(self.train_loader):
            data = data_batch[0].to(self.device)
            landmarks_gt = data_batch[1].to(self.device)
            hand_presence_gt = data_batch[2].to(self.device)
            handedness_gt = data_batch[3].to(self.device)

            self.optimizer.zero_grad()
            landmarks_pred, hand_presence_pred, handedness_pred = self.model(
                data
            )
            regression_loss = self.criterion[0](landmarks_pred, landmarks_gt)
            regression_loss.backward(retain_graph=True)
            handedness_loss = self.criterion[1](handedness_pred, handedness_gt)
            handedness_loss.backward(retain_graph=True)
            hand_presence_loss = self.criterion[2](
                hand_presence_pred, hand_presence_gt
            )
            hand_presence_loss.backward(retain_graph=True)

            regression_losses.update(regression_loss.item(), data.size(0))
            handedness_losses.update(handedness_loss.item(), data.size(0))
            hand_presence_losses.update(
                hand_presence_loss.item(), data.size(0)
            )

            self.optimizer.step()

            if i % self.print_freq == 0:
                log = """Epoch: {0}/{1}\t
                [{2}/{3}]\t
                regression_loss {4:.3f}\t
                hand_presence_loss {5:.3f}\t
                handedness_loss {6:.3f}
                """.format(
                    epoch + 1,
                    self.epochs,
                    i + 1,
                    len(self.train_loader),
                    regression_losses.avg,
                    hand_presence_losses.avg,
                    handedness_losses.avg,
                ).replace(
                    "\n", ""
                )
                print(log, end="\r")
                print(" " * len(log.expandtabs()), end="\r")

        self.regression_train_loss.append(regression_losses.avg)
        self.hand_presence_train_loss.append(hand_presence_losses.avg)
        self.handedness_train_loss.append(handedness_losses.avg)

    def val_epoch(self, epoch):
        regression_losses = AverageMeter()
        handedness_losses = AverageMeter()
        hand_presence_losses = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(self.val_loader):
                data = data_batch[0].to(self.device)
                landmarks_gt = data_batch[1].to(self.device)
                hand_presence_gt = data_batch[2].to(self.device)
                handedness_gt = data_batch[3].to(self.device)

                (
                    landmarks_pred,
                    hand_presence_pred,
                    handedness_pred,
                ) = self.model(data)
                regression_loss = self.criterion[0](
                    landmarks_pred, landmarks_gt
                )
                handedness_loss = self.criterion[1](
                    handedness_pred, handedness_gt
                )
                hand_presence_loss = self.criterion[1](
                    hand_presence_pred, hand_presence_gt
                )

                regression_losses.update(regression_loss.item(), data.size(0))
                handedness_losses.update(handedness_loss.item(), data.size(0))
                hand_presence_losses.update(
                    hand_presence_loss.item(), data.size(0)
                )

                if i % self.print_freq == 0:
                    log = """Epoch: {0}/{1}\t
                    [{2}/{3}]\t
                    regression_val_loss {4:.3f}\t
                    hand_presence_val_loss {5:.3f}\t
                    handedness_val_loss {6:.3f}
                    """.format(
                        epoch + 1,
                        self.epochs,
                        i,
                        len(self.val_loader),
                        regression_losses.avg,
                        hand_presence_losses.avg,
                        handedness_losses.avg,
                    ).replace(
                        "\n", ""
                    )
                    print(log, end="\r")
                    print(" " * len(log.expandtabs()), end="\r")

        log = """Epoch: {0}/{1}\t
        [{2}/{3}]\t
        regression_val_loss {4:.3f}\t
        hand_presence_val_loss {5:.3f}\t
        handedness_val_loss {6:.3f}
        """.format(
            epoch + 1,
            self.epochs,
            i,
            len(self.val_loader),
            regression_losses.avg,
            hand_presence_losses.avg,
            handedness_losses.avg,
        ).replace(
            "\n", ""
        )

        print(log)

        self.regression_val_loss.append(regression_losses.avg)
        self.hand_presence_val_loss.append(hand_presence_losses.avg)
        self.handedness_val_loss.append(handedness_losses.avg)

        if self.scheduler:
            self.scheduler.step(self.regression_val_loss[-1])

        self.save_checkpoint(epoch, regression_losses.avg)

    def save_checkpoint(self, epoch, loss):
        checkpoint_save_path = os.path.join(
            os.path.dirname(self.saved_path),
            "{}_checkpoint.pt".format(
                os.path.splitext(os.path.basename(self.saved_path))[0]
            ),
        )

        best_checkpoint_path = os.path.join(
            os.path.dirname(self.saved_path),
            "{}_checkpoint_best.pt".format(
                os.path.splitext(os.path.basename(self.saved_path))[0]
            ),
        )
        self.best_checkpoint_path = best_checkpoint_path

        os.makedirs(os.path.dirname(checkpoint_save_path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler,
                "criterion": self.criterion,
                "history": self.history,
                "min_loss": loss,
            },
            checkpoint_save_path,
        )

        if loss < self.min_regression_loss:
            copyfile(checkpoint_save_path, best_checkpoint_path)
            wandb.save(checkpoint_save_path)
            logger.info(
                "sucessfully saved best model with smallest loss of {min_loss}"
            )
            self.min_regression_loss = loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
