# STUDENT's UCO: 525265

# Description:
# This file should be used for performing inference on a network
# Usage: inference.py <dataset_path> <model_path>

from argparse import ArgumentParser
from pathlib import Path

import torch
from training import config, transforms
from dataset import read_inference, CityScapeDataset
import pandas as pd

# declaration for this function should not be changed
@torch.no_grad()  # do not calculate the gradients
def inference(dataset_path: Path, model_path: Path, reference: bool = False) -> None:
    """Performs inference on the given dataset using the specified model.

    Args:
        dataset_path: Path to the dataset. The function processes
                      all PNG images in this directory (optionally
                      recursively in its subdirectories).
        model_path: Path to the model file.

    Saves:
        predictions to 'output_predictions' folder.
        The files can be saved in a flat structure with
        the same name as the input file.
    """
    # Check for available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computing with {}!".format(device))

    # loading the model
    model = config["net"](config["n_classes"])
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    df = read_inference(dataset_path)
    predict_dataset = CityScapeDataset(df, transforms, {"inference": 0})
    predict_dl = torch.utils.data.DataLoader(predict_dataset, batch_size=1)

    print("Inferencing!!")
    results = []
    i = 0
    for image, _ in predict_dl:
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        results.append(predicted[0].item())
        i += 1

    save_df = pd.DataFrame()
    save_df["filename"] = df["img_path"]
    save_df["class_id"] = pd.Series(results)
    output_path = Path("output_predictions") / Path("predictions.csv")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    save_df.to_csv(str(output_path))

    if not reference:
        return

    i = 0
    results = []
    for _, true_label in predict_dl:
        results.append(true_label.item())
        i += 1
    
    save_df = pd.DataFrame()
    save_df["filename"] = df["img_path"]
    save_df["class_id"] = pd.Series(results)
    save_df.to_csv(Path("reference_predictions") / Path("true_labels.csv"))


# #### code below should not be changed ############################################################################
def main() -> None:
    parser = ArgumentParser(description="Inference script for a neural network.")
    parser.add_argument("dataset_path", type=Path,
                        help="Path to the dataset")
    parser.add_argument("model_path", type=Path,
                        help="Path to the model weights")
    args = parser.parse_args()
    inference(args.dataset_path, args.model_path)


if __name__ == "__main__":
    main()
