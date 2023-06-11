import os
import sys
import json
import csv
import argparse
import datetime
import subprocess
import time
import threading

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from gender_model import GenderModel
from torch.utils.tensorboard import SummaryWriter


def parse_command_line_args():
    parser = argparse.ArgumentParser(
        description="Torch-based Gender Detection workflow and workbench",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "test", "inference"],
        help="Workflow scheme [train | test | inference]",
    )

    parser.add_argument(
        "conf", type=str, help="Path to configuration file defining an experiment"
    )

    parser.add_argument(
        "--embedding_pretrained",
        type=str,
        help="Name shared by the .model and .conf files of a pretrained embedding model",
    )

    parser.add_argument(
        "--history",
        type=str,
        help="Path of the history folder",
    )

    parser.add_argument(
        "--finetune",
        type=str,
        help="Whether to limit retraining to last layer",
    )

    return parser.parse_args()


def load_json(json_file):
    with open(json_file, "r") as file:
        jdata = json.load(file)
    return jdata


def display_params(args, params):
    print("SCRIPT: " + os.path.basename(__file__))
    print("Options...")
    for arg in vars(args):
        print("  " + arg + ": " + str(getattr(args, arg)))
    print("-" * 30)

    print("Config-file params...")
    for key, value in params.items():
        print("  " + key + ": " + str(value))
    print("-" * 30)


def get_time():
    return datetime.datetime.now()


def _main_():
    args = parse_command_line_args()
    params = load_json(args.conf)
    display_params(args, params)

    model = GenderModel(args.conf, args.embedding_pretrained)

    if args.finetune:
        print(f"\nLoading weights from {args.finetune}...\n")
        checkpoint = torch.load(args.finetune)
        model_state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(model_state_dict)
        print("Loading finished...\n")

    writer = SummaryWriter(args.history)

    if args.mode == "train":
        print("Running tensorboard...\n")

        def run_command(command):
            subprocess.call(command, shell=True)

        thread = threading.Thread(
            target=run_command, args=(f"tensorboard --logdir={args.history}",)
        )
        thread.start()

        time.sleep(5)

        data = []
        with open("aio_data.csv", "r", encoding="utf-8") as reader:
            reader = csv.reader(reader)
            for row in reader:
                data.append(row)

        header = data.pop(0)
        dataset_size = len(data)
        split = int(dataset_size * 0.8)

        train_data, val_data = random_split(data, [split, dataset_size - split])
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)
        save_every = 20

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(model.parameters(), lr=0.0004)

        print("\nInitialization Finished...\n")
        for epoch in range(100):
            print(f"Epoch {epoch+1}/100: [", end="")
            for iteration, (inputs, counts, labels) in enumerate(train_loader):
                print("=", end="", flush=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                labels = torch.tensor([1 if gd == "m" else 0 for gd in list(labels)])
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                writer.add_scalar(
                    f"Train_loss in {epoch}th epoch", loss.item(), iteration
                )
                writer.flush()

            writer.add_scalar("Train_loss", loss.item(), epoch)

            if epoch % save_every == 0:
                checkpoint_path = f"./Checkpoint/checkpoint_{get_time()}_{epoch}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    checkpoint_path,
                )

            with torch.no_grad():
                val_loss = 0
                total = 0
                correct = 0
                for i, (val_inputs, val_counts, val_labels) in enumerate(val_loader):
                    val_outputs = model(val_inputs)
                    val_labels = torch.tensor(
                        [1 if gd == "m" else 0 for gd in list(val_labels)]
                    )
                    val_loss += criterion(val_outputs, val_labels).item()
                    val_recognize = torch.max(val_outputs, 1)
                    total += val_labels.size(0)
                    correct += (val_recognize.indices == val_labels).sum().item()

                val_loss = val_loss / len(val_loader)
                writer.add_scalar("Validation_loss", loss.item(), epoch)
                writer.add_scalar("Validation_acc", correct / total * 100, epoch)

            writer.flush()
            print("]")


if __name__ == "__main__":
    _main_()
