# STUDENT's UCO: 525265

# Description:
# This file should contain custom dataset class.
# The class should subclass the torch.utils.data.Dataset.

import torch
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from skimage import io


class SampleDataset(Dataset[Tensor]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tensor:
        return torch.zeros((256, 256))


def read_City_Scape(labels_path: Path):
    '''
    Reads labels of CityScape dataset in the original format.
    Returns labels in a Dataframe.
    '''
    # get all label directories
    label_dirs = [dir for dir in labels_path.iterdir()]

    # iterate over label files
    samples = []
    for label in label_dirs:
        if not label.is_dir():
            continue

        print(f"Read {str(label)}")
        for image in label.iterdir():
            img_path = image
            if not img_path.exists():
                continue
            samples.append((str(img_path), label.name))

    return pd.DataFrame(samples, columns=['img_path', 'label'])


def read_inference(read_path: Path):
    '''
    Reads labels of CityScape dataset in the original format.
    Returns labels in a Dataframe.
    '''
    samples = []
    for image in read_path.iterdir():
        if not image.exists():
            continue
        if image.is_dir():
            res = read_inference(image)
            img_paths = res["img_path"].to_list()
            labels = res["label"].to_list()
            samples += [(img_paths[i], labels[i]) for i in range(len(img_paths))] 
        elif str(image).split(".")[-1] != "txt":
            samples.append((str(image), "inference"))

    return pd.DataFrame(samples, columns=['img_path', 'label'])


class CityScapeDataset(Dataset[Tensor]):
    def __init__(self, dataset_df, transforms, label_map):
        self.samples_df = dataset_df
        self.transform = transforms
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.samples_df)

    def __getitem__(self, idx: int) -> Tensor:
        sample = self.samples_df.iloc[idx]
        # print(sample["img_path"])
        img = io.imread(sample['img_path'])

        transformed = self.transform(image=img)
        return transformed["image"], self.label_map[sample["label"]]


def get_dataset_stats(ds):
    '''
    Inputs:
        ds   - pytorch dataset
             - the sample shape [3, W, H]
             - torch tensor, values in range (0, 1)

    Outputs:
        dataset mean value per color channel
        dataset standard deviation per color channel
    '''
    x = torch.stack([sample for sample, _ in ds])
    ds_mean = x.float().mean(axis=(0, 2, 3))
    ds_std = x.float().std(axis=(0, 2, 3))
    ds_max = x.float().max()
    return ds_mean, ds_std, ds_max
