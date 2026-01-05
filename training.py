# STUDENT's UCO: 525265

# Description:
# This file should be used for performing training of a network
# Usage: python training.py <dataset_path>

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchview import draw_graph

from dataset import CityScapeDataset, read_City_Scape, get_dataset_stats
from network import Alexnet, PretrainedAlexnet, PretrainedResnet, PretrainedVGG, Resnet

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import numpy as np


# Configuration File
config = {
        'batch_size': 64,
        'num_workers': 0,
        'dropout': 0.5,
        'lr': 0.0001,
        'optimizer': torch.optim.Adam,
        'img_size': 256,
        'n_classes': 6,
        "epochs": 200,
        "patience": 10,
        "tolerance": 0.02,
        # "net": lambda num_classes: PretrainedResnet(num_classes=num_classes),
        # "net": lambda num_classes: Alexnet(num_classes=num_classes,
        #                                   dropout=config["dropout"]),
        # "net": lambda num_classes: PretrainedAlexnet(num_classes=num_classes,
        #                                               dropout=config["dropout"]),
        # "net": lambda num_classes: PretrainedVGG(num_classes=num_classes,
        #                                           dropout=config["dropout"]),
        "net": lambda num_classes: Resnet(num_classes=num_classes)
    }

transforms_train = A.Compose(
    [
        # augmentation
        A.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.1),
        A.GaussNoise(),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(rotate_limit=60, p=0.8, border_mode=0),
        A.RGBShift(r_shift_limit=45, g_shift_limit=30, b_shift_limit=45, p=0.7),

        # preprocessing
        A.SmallestMaxSize(config['img_size']),
        A.CenterCrop(config['img_size'], config['img_size']),
        A.Normalize(mean=torch.tensor([72.0531, 54.4551, 64.2774])/255.0,
                    std=torch.tensor([48.4120, 40.9302, 44.7423])/255.0,
                    max_pixel_value=255.0, p=1),
        ToTensorV2(),
    ])

transforms = A.Compose(
    [
        # preprocessing
        A.SmallestMaxSize(config['img_size']),
        A.CenterCrop(config['img_size'], config['img_size']),
        A.Normalize(mean=torch.tensor([72.0531, 54.4551, 64.2774])/255.0,
                    std=torch.tensor([48.4120, 40.9302, 44.7423])/255.0,
                    max_pixel_value=255.0, p=1),
        ToTensorV2(),
    ])


# sample function for model architecture visualization
# draw_graph function saves an additional file: Graphviz DOT graph file,
# it's not necessary to delete it
def draw_network_architecture(
            net: torch.nn.Module, input_sample: torch.Tensor
                              ) -> None:
    # saves visualization of model architecture to the model_architecture.png
    draw_graph(
        net,
        input_sample,
        graph_dir="TB",
        save_graph=True,
        filename="model_architecture",
        expand_nested=True,
    )


# sample function for losses visualization
def plot_learning_curves(
    train_losses: list[float], validation_losses: list[float]
) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses During Training")
    plt.plot(train_losses, label="train_loss")
    plt.plot(validation_losses, label="validation_loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curves.png")


def accuracy(net, data_loader, dev):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(dev), labels.to(dev)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def loss_bch(model, loss_func, xb, yb, dev, opt=None):
    xb, yb = xb.to(dev), yb.to(dev)
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def train(net, train_dataloader, loss_func, device, optimizer):
    net.train()
    loss, size = 0, 0
    for _, (xb, yb) in tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader), leave=False):
        b_loss, b_size = loss_bch(net, loss_func, xb, yb, device, optimizer)
        loss += b_loss * b_size
        size += b_size

    return loss / size


def validate(net, val_dataloader, loss_func, device, optimizer=None):
    net.eval()
    with torch.no_grad():
        losses, nums = zip(
            *[loss_bch(net, loss_func, xb, yb, device)
              for xb, yb in val_dataloader]
        )

    return np.sum(np.multiply(losses, nums)) / np.sum(nums)


class BestModel_Callback:
    def __init__(self, model_path='best_model.pt'):
        self.best_valid_loss = float('inf')
        self.model_path = model_path

    def __call__(self, model, valid_loss):
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            torch.save(model.state_dict(), self.model_path)


class EarlyStopping_Callback:
    def __init__(self, patience=5, tolerance=0):
        self.patience = patience
        self.tolerance = tolerance
        self.best_valid_loss = float('inf')
        self.increasing_steps = 0

    def __call__(self, valid_loss):
        if valid_loss > self.best_valid_loss + self.tolerance:
            self.increasing_steps += 1
        else:
            self.increasing_steps = 0

        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss

        return self.increasing_steps >= self.patience


# sample function for training
def fit(
    net: torch.nn.Module,
    epochs: int,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    patience: int,
    tolerance: float,
) -> tuple[list[float], list[float]]:

    train_losses: list[float] = []
    val_losses: list[float] = []
    early_stop = EarlyStopping_Callback(patience=patience, tolerance=tolerance)
    best_model = BestModel_Callback("best_model.pt")

    for epoch in tqdm(range(epochs)):
        train_loss = train(net, train_dataloader, loss, device, optimizer)
        val_loss = validate(net, val_dataloader, loss, device)

        print(f'\nepoch {epoch+1}/{epochs}, loss: {train_loss:.05f},'
              f'validation loss: {val_loss:.05f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        best_model(net, val_loss)
        if early_stop(val_loss):
            print("Early stopping")
            break

    net.load_state_dict(torch.load("best_model.pt"))
    print("Training finished!")
    return train_losses, val_losses


# declaration for this function should not be changed
def training(dataset_path: Path) -> None:
    """Performs training on the given dataset.

    Args:
        dataset_path: Path to the dataset.

    Saves:
        - model.pt (trained model)
        - learning_curves.png (learning curves generated during training)
        - model_architecture.png (a scheme of model's architecture)
    """

    # Check for available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computing with {}!".format(device))

    # Loading Datasets and computing its stats
    pd_dataset = read_City_Scape(dataset_path)
    label_map = {
        label: key for key, label in
        enumerate(sorted(pd_dataset["label"].unique().astype(str)))
                }
    train_df, test_df = train_test_split(pd_dataset, test_size=0.2)
    train_df, valid_df = train_test_split(train_df, test_size=0.2)

    # Computing stats
    stats_transform = A.Compose([
                         # preprocessing
                         A.SmallestMaxSize(config['img_size']),
                         A.CenterCrop(config['img_size'], config['img_size']),
                         ToTensorV2(),
                        ]
                    )
    mean, std, max_val = get_dataset_stats(
        CityScapeDataset(train_df, stats_transform, label_map)
                                  )
    print(f'stats\n  mean:\t {mean}\n  std:\t {std}\n'
          f'max_val:\t {max_val}')

    # Creating Dataloaders
    train_dataset = CityScapeDataset(train_df, transforms_train, label_map)
    val_dataset = CityScapeDataset(valid_df, transforms, label_map)
    test_dataset = CityScapeDataset(test_df, transforms, label_map)
    train_dataloader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=config["batch_size"]
                        )
    val_dataloader = torch.utils.data.DataLoader(
                        val_dataset, batch_size=config["batch_size"]
                        )
    test_dataloader = torch.utils.data.DataLoader(
                        test_dataset, batch_size=config["batch_size"]
                        )

    # Drawing Graph
    net = config["net"](config["n_classes"])
    input_sample = torch.zeros((1, 3, config["img_size"], config["img_size"]))
    draw_network_architecture(net, input_sample)
    net.to(device)

    optimizer = config["optimizer"](net.parameters(), lr=config['lr'])
    loss = torch.nn.functional.cross_entropy

    # Let's Train!!
    train_losses, val_losses = fit(
        net, config["epochs"], train_dataloader, val_dataloader,
        loss, optimizer, device, config["patience"], config["tolerance"]
    )
    print(f'accuracy train {accuracy(net, train_dataloader, device)}\n')
    print(f'accuracy valid {accuracy(net, val_dataloader, device)}\n')
    print(f'accuracy test {accuracy(net, test_dataloader, device)}\n')

    # Let's Save!
    torch.save(net.state_dict(), "model.pt")
    plot_learning_curves(train_losses, val_losses)


# #### code below should not be changed #######################
def main() -> None:
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()
    training(args.dataset_path)


if __name__ == "__main__":
    main()
