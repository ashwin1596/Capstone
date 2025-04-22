import torch
from torch import nn
import torch.nn.utils.prune as prune

from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
import torchvision.transforms as transforms

import mlflow

import os
import sys

sys.path.append(os.path.abspath(".."))
from passport_model import PassportModel

checkpoint_dir = "fine_tuning_checkpoints"
latest_checkpoint = os.path.join(checkpoint_dir, "latest_model.pth")
best_checkpoint = os.path.join(checkpoint_dir, "best_model.pth")

# Ensure checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

# Get cpu or gpu for training.
device = "cuda:3" if torch.cuda.is_available() else "cpu"

# Define the normalization transform
normalize_transform = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

# Load the dataset with custom transformation
transform = transforms.Compose([
    ToTensor(),
    normalize_transform
])

training_data = datasets.CIFAR10(
    root="../data",
    train=True,
    download=True,
    transform=transform,
)

test_data = datasets.CIFAR10(
    root="../data",
    train=False,
    download=True,
    transform=transform,
)

# Split the training data into a training and validation dataset.
train_size = int(0.8 * len(training_data))
val_size = len(training_data) - train_size
training_data, val_data = torch.utils.data.random_split(training_data, [train_size, val_size])

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

def prune_model(model, parameter):
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name=parameter, amount=0.2)
            prune.remove(module, name=parameter)
        elif isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name=parameter, amount=0.4)
            prune.remove(module, name=parameter)

def evaluate(dataloader, model, loss_fn, metrics_fn, epoch, phase="Validation"):
    """Evaluate the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        epoch: an integer, the current epoch number.
    """
    num_batches = len(dataloader)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred, _ = model(X)
            eval_loss += loss_fn(pred, y).item()
            eval_accuracy += metrics_fn(pred, y)

    eval_loss /= num_batches
    eval_accuracy /= num_batches
    mlflow.log_metric(f"{phase.lower()}_loss", f"{eval_loss:2f}", step=epoch)
    mlflow.log_metric(f"{phase.lower()}_accuracy", f"{eval_accuracy:2f}", step=epoch)

    print(f"{phase} metrics: \nAccuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} \n")

    return eval_loss, eval_accuracy

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device="cuda:3" if torch.cuda.is_available() else "cpu"):
    """Loads model, optimizer, and scheduler states from a checkpoint file."""

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")

    required_keys = ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch", "loss"]

    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint is missing key: {key}")


    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Resume from the next epoch
    best_val_loss = checkpoint["loss"]  # Restore best validation loss

    print(f"Loaded checkpoint from '{checkpoint_path}' at epoch {start_epoch} with loss {best_val_loss:.4f}")
    return start_epoch, best_val_loss

def train(dataloader, model, loss_fn, metrics_fn, optimizer, epoch):
    """Train the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the training data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        optimizer: an instance of `torch.optim.Optimizer`, the optimizer used for training.
        epoch: an integer, the current epoch number.
    """
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred, _ = model(X)
        loss = loss_fn(pred, y)
        accuracy = metrics_fn(pred, y)

        # Backpropagation.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            step = batch // 100 * (epoch + 1)
            mlflow.log_metric("loss", f"{loss:2f}", step=step)
            mlflow.log_metric("accuracy", f"{accuracy:2f}", step=step)
            print(f"loss: {loss:2f} accuracy: {accuracy:2f} [{current} / {len(dataloader)}]")

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("/cifar10_sig_enc_wm_final")

with mlflow.start_run() as run:
    mlflow.run_name = "Pruning and Fine-tuning"

    # Load the model 
    logged_model = "runs:/9ff10be238c648418ea638b65529c4ce/sign_enc_model"
    loaded_model = mlflow.pytorch.load_model(logged_model)
    loaded_model.to(device)
    to_prune_model = mlflow.pytorch.load_model(logged_model)
    to_prune_model.to(device)

    # Prune the model
    prune_model(to_prune_model, "weight")

    # Log the pruned model
    mlflow.pytorch.log_model(to_prune_model, "pruned_model")

    print("Model pruned successfully!")

    # Fine-tune the pruned model
    epochs = 20
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
    optimizer = torch.optim.Adam(loaded_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    start_epoch = 0  # Default starting epoch
    best_val_loss = float("inf")  # Default best evaluation loss

    for t in range(start_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, loaded_model, loss_fn, metric_fn, optimizer, epoch=t)
        val_loss, val_accuracy = evaluate(val_dataloader, loaded_model, loss_fn, metric_fn, epoch=t)

        scheduler.step(val_loss)

        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metric("learning_rate", current_lr, step=t)

        # save latest model
        latest_checkpoint = "fine_tuning_checkpoints/latest_model.pth"
        torch.save({
            "epoch": t,
            "model_state_dict": loaded_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": val_loss
        }, latest_checkpoint)

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = "fine_tuning_checkpoints/best_model.pth"
            torch.save({
                "epoch": t,
                "model_state_dict": loaded_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": val_loss
            }, best_checkpoint)

            mlflow.log_metric("best_val_loss", val_loss)
            mlflow.log_metric("best_val_accuracy", val_accuracy)
            mlflow.log_metric("best_epoch", t)
        
            mlflow.log_artifact(best_checkpoint)

            # Save the best model state dict
            best_model_state_dict = loaded_model.state_dict()


    # Evaluate the best model on the test dataset
    test_loss, test_accuracy = evaluate(test_dataloader, loaded_model, loss_fn, metric_fn, epoch=epochs, phase="Test")

    if best_model_state_dict is not None:
        loaded_model.load_state_dict(best_model_state_dict)
        mlflow.pytorch.log_model(loaded_model, "finetuned_model")