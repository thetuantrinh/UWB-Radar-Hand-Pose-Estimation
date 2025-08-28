from torch import nn


class ComputeLoss:
    def __init__(self):
        self.loss = 0
        self.regression_cri = nn.L1Loss()
        self.classification_cri = nn.BCELoss()

    def __call__(self, output, target):
        regression_loss = self.regression_cri(output, target)
        classification_loss = self.classification_cri(output, target)

        return (regression_loss + classification_loss)
