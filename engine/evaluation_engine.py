import torch


class ValidationEngine:
    def __init__(
        self,
        model,
        val_loader,
        criterion,
    ):
        self.model = model
        self.val_loader = val_loader
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.criterion = criterion.to(self.device)
        self.model.to(self.device)
        self.print_freq = 10

    def evaluate(self):
        losses = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, val_batch in enumerate(self.val_loader):
                data = val_batch[0].to(self.device)
                target = val_batch[1].to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                losses.update(loss.item(), data.size(0))

                if i % self.print_freq == 0:
                    print(
                        "Test: [{0}/{1}]\t"
                        "val_loss {2:.3f}".format(
                            i,
                            len(self.val_loader),
                            losses.avg,
                        ),
                        end="\r",
                    )
        print(
            "Test: [{0}/{1}]\t"
            "val_loss {2:.3f}".format(
                i, len(self.val_loader), losses.avg
            )
        )


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
