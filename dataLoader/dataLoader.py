from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
from typing import Tuple
from torchvision import transforms as T


def load_data(path_to_data_folder: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Loads the data from the given path.

    :param path_to_data_folder: the path to the data folder
    :return: a tuple of two DataLoaders, one for the training data and one for the test data
    """
    BATCH_SIZE = batch_size
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data = ImageFolder(path_to_data_folder, transform=transform)
    train_data, val_data = torch.utils.data.random_split(data,
                                                         [int(0.8 * len(data)), len(data) - int(0.8 * len(data))])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = load_data('../data/')
    print(len(train_loader))
    print(len(val_loader))
