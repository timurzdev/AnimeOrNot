from torch.utils.data import DataLoader, Dataset
import torch
from typing import Tuple
from torchvision import transforms as T
import pandas as pd
import PIL
from torchvision.datasets import ImageFolder
from typing import Dict


class AnimeHumansDataset(Dataset):
    def __init__(self, annotations_path: str, data_transform: T.Compose = None):
        self.data_transform = data_transform
        self.df = pd.read_csv(annotations_path)
        self.label_shape = (2, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label = torch.zeros(self.label_shape, dtype=torch.float)
        label[self.df.iloc[idx, 1]] = 1
        label = label.squeeze()
        img = PIL.Image.open(img_path)
        if self.data_transform is not None:
            img = self.data_transform(img)
        sample = {'image': img, 'label': label}
        return sample


def load_data(path_to_data_folder: str, batch_size: int) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Loads the data from the given path.

    :param path_to_data_folder: the path to the data folder
    :return: a tuple of two DataLoaders, one for the training data and one for the test data
    """
    BATCH_SIZE = batch_size
    transform = T.Compose(
        [
            T.Resize((128, 128)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    data = ImageFolder(path_to_data_folder, transform=transform)
    classes = data.classes
    print(data.class_to_idx)
    print(classes)
    train_data, val_data = torch.utils.data.random_split(data,
                                                         [int(0.8 * len(data)), len(data) - int(0.8 * len(data))])
    image_datasets = {'train': train_data, 'val': val_data}
    data_loaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, num_workers=4, shuffle=True) for x in
        image_datasets
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return data_loaders['train'], data_loaders['val'], dataset_sizes


if __name__ == '__main__':
    train_loader, val_loader, dataset_sizes = load_data('../data/', 16)
    print(len(train_loader))
    print(len(val_loader))
    data, label = next(iter(train_loader))
    print(label.shape)
