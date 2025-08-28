from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch


def split_dataset(
    dataset,
    batch_size,
    train_ratio=0.8,
    shuffle=True,
    seed=42,
    workers=4,
    pin_memory=True,
):
    dataset_size = len(dataset)
    indices = np.random.permutation(dataset_size)
    # indices = list(range(dataset_size))
    split = int(np.floor(train_ratio * dataset_size))
    if shuffle:
        np.random.seed(seed=seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def create_batch_loader(
    data: DataLoader,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=4,
):
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )


def get_mean_and_std(train_loader: DataLoader):
    print("getting mean and std of the data set")
    mean = torch.zeros((train_loader.dataset[0][0].shape))
    std = torch.zeros((train_loader.dataset[0][0].shape))
    num_samples = 0

    for sample in train_loader:
        data = sample[0]
        batch_samples = data.size(0)
        # data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(0)
        std += data.std(0)
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples

    return [mean, std]
