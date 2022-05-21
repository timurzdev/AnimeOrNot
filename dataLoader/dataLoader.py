from torch.utils.data import DataLoader, Dataset
import torch
from typing import Tuple
from torchvision import transforms as T
import pandas as pd
import PIL
from typing import Dict


class AnimeHumansDataset(Dataset):
    def __init__(self, annotations_path: str, data_transform: T.Compose = None):
        self.data_transform = data_transform
        self.annotations_path = annotations_path
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


def load_data(path_to_data_folder: str, batch_size: int) \
        -> Tuple[Dict, Dict]:
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
    train_dataset = AnimeHumansDataset(annotations_path=path_to_data_folder + 'anime_humans.csv',
                                       data_transform=transform)
    test_data = AnimeHumansDataset(annotations_path=path_to_data_folder + 'anime_humans_test.csv',
                                   data_transform=transform)
    train_data, val_data = torch.utils.data.random_split(train_dataset,
                                                         [int(0.8 * len(train_dataset)),
                                                          len(train_dataset) - int(0.8 * len(train_dataset))])
    image_datasets = {'train': train_data, 'val': val_data, 'test': test_data}
    data_loaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, num_workers=4, shuffle=True) for x in
        ['train', 'val', 'test']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    return data_loaders, dataset_sizes


if __name__ == '__main__':
    data_loaders, dataset_sizes = load_data('../data/', 16)
    train_loader = data_loaders['train']
    data = next(iter(train_loader))
    input, label = data['image'], data['label']
    print(label.max(), label.min())
