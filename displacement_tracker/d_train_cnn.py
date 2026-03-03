from __future__ import annotations
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import yaml
import datetime
from displacement_tracker.paired_image_dataset import PairedImageDataset
from displacement_tracker.simple_cnn import SimpleCNN
from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("train-cnn")


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.data = [base_ds[i] for i in range(len(base_ds))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def custom_collate(batch):
    # Initialize an empty dictionary to store concatenated results
    collated_dict = {}

    # Iterate over keys in the first dictionary of the batch
    for key in batch[0].keys():
        entry = [d[key] for d in batch]

        if key != "meta":
            entry = torch.stack(entry, dim=0)

        collated_dict[key] = entry

    return collated_dict


@click.command()
@click.argument("config", type=click.Path(exists=True))
def cli(config: str) -> None:
    with open(config, "r") as f:
        params = yaml.safe_load(f)
    required = ["hdf5", "training"]
    for k in required:
        if k not in params:
            raise click.ClickException(f"Missing required config key: {k}")
    train(params["hdf5"], **params["training"])


def train(
    hdf5_path: str,
    training_frac: float,
    validation_frac: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    sigma: float = 3.0,
    checkpoint: str | None = None,
    device: str | None = None,
    model_kwargs: dict | None = None,
    memory: bool = False,
) -> None:

    model_kwargs = model_kwargs or {}

    # Set device to GPU if available
    if torch.cuda.is_available() and (device is None or device == "cuda"):
        device = torch.device("cuda")
        LOGGER.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif device is None or device == "cpu":
        device = torch.device("cpu")
        LOGGER.info("Using CPU")
    else:
        raise Exception(f"Could not find device {device}")

    if checkpoint:
        checkpoint = Path(checkpoint)
        LOGGER.warning("Loading model from checkpoint, ignoring model_kwargs.")
        model = SimpleCNN.from_pth(checkpoint).to(device)

        save_loc = checkpoint.parent
    else:
        model = SimpleCNN(3, 1, **model_kwargs).to(device)
        save_loc = None

    # Load and shuffle dataset
    dataset = PairedImageDataset(hdf5_path, sigma=sigma)
    splits = [training_frac, validation_frac, 1 - training_frac - validation_frac]

    (train_set, val_set, _), idcs_list = dataset.create_subsets(
        splits, shuffle=True, save_loc=save_loc
    )

    if memory:
        LOGGER.info("Caching training dataset in RAM...")
        train_set = CachedDataset(train_set)
        LOGGER.info(f"Cached {len(train_set)} training samples.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=custom_collate,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=custom_collate)

    LOGGER.info(
        f"Split {len(dataset)} samples into {len(train_set)} train and {len(val_set)} validation samples."
    )

    def criterion(x, y):
        return torch.nn.functional.mse_loss(x, y)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create timestamped run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # caching splits for future use
    with open(os.path.join(run_dir, "splits.csv"), "w") as split_file:
        split_file.write(",".join([str(split) for split in splits]) + "\n")
        for idcs in idcs_list:
            split_file.write(",".join([str(idx) for idx in idcs]) + "\n")

    best_eval = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train = 0

        for i, entry in enumerate(train_loader):
            diff = entry["feature"] - entry["prewar"]
            feats = torch.cat((entry["feature"], entry["prewar"], diff), axis=1).to(
                device
            )
            labels = entry["label"].to(device)

            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bsz = labels.size(0)
            train_loss += loss.item() * bsz
            n_train += bsz
            print(
                f"Epoch {epoch}: Completed step {i + 1} / {len(train_loader)}", end="\r"
            )

        # normalize once per epoch
        train_loss /= n_train

        # Validation loss
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for entry in val_loader:
                diff = entry["feature"] - entry["prewar"]
                feats = torch.cat((entry["feature"], entry["prewar"], diff), axis=1).to(
                    device
                )
                labels = entry["label"].to(device)

                outputs = model(feats)
                loss = criterion(outputs, labels)

                bsz = labels.size(0)
                val_loss += loss.item() * bsz
                n_val += bsz

        val_loss = val_loss / n_val if n_val > 0 else 0.0

        if val_loss < best_eval:
            model_path = os.path.join(run_dir, "best_model.pth")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_args": getattr(model, "config", {}),
                },
                model_path,
            )
            LOGGER.info(
                f"Best model saved to {model_path} at epoch {epoch} with loss {val_loss} < {best_eval}."
            )
            best_eval = val_loss
        LOGGER.info(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}"
        )


if __name__ == "__main__":
    cli()
