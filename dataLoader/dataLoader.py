from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch
from typing import Tuple


def load_data(path_to_data_folder: str) -> Tuple[DataLoader, DataLoader]:
    """
    Loads the data from the given path.

    :param path_to_data_folder: the path to the data folder
    :return: a tuple of two DataLoaders, one for the training data and one for the test data
    """
    data = ImageFolder(path_to_data_folder)
    train_data, val_data = torch.utils.data.random_split(data,
                                                          [int(0.8 * len(data)), len(data) - int(0.8 * len(data))])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = load_data('../data/')
    print(len(train_loader))
    print(len(val_loader))

