import torch
from torch import nn
import torch.nn.utils.prune as prune

from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
import torchvision.transforms as transforms
import torch.nn.functional as F
import mlflow
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature

import numpy as np

import os
import sys

sys.path.append(os.path.abspath(".."))
from models import BaseModel, BaseModel2, StudentModel

checkpoint_dir = "distillation_checkpoints"
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

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for balancing soft and hard targets
        self.temperature = temperature  # Temperature for softening probability distributions
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Hard loss: Cross-entropy between student logits and ground truth labels
        hard_loss = self.cross_entropy(student_logits, labels)

        # Soft loss: KL divergence between softened teacher and student distributions
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * self.temperature**2

        # combine loss
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return loss

def train(teacher_model, student_model, dataloader, loss_fn, optimizer, metrics_fn, epoch):
    teacher_model.eval()

    student_model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass for both models
        with torch.no_grad():
            teacher_logits = teacher_model(X)
            
        student_logits = student_model(X)

        # Calculate distillation loss
        loss = loss_fn(student_logits, teacher_logits, y)
        
        accuracy = metrics_fn(student_logits, y)

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
    
    return student_model

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

def evaluate(teacher_model, student_model, dataloader, loss_fn, metrics_fn, epoch, phase="Validation"):
    """Evaluate the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        epoch: an integer, the current epoch number.
    """
    num_batches = len(dataloader)
    student_model.eval()
    teacher_model.eval()
    eval_loss, eval_accuracy = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Get teacher and student outputs
            teacher_logits = teacher_model(X)
            student_logits = student_model(X)

            eval_loss += loss_fn(student_logits, teacher_logits, y).item()
            eval_accuracy += metrics_fn(student_logits, y)

    eval_loss /= num_batches
    eval_accuracy /= num_batches
    mlflow.log_metric(f"{phase.lower()}_loss", f"{eval_loss:2f}", step=epoch)
    mlflow.log_metric(f"{phase.lower()}_accuracy", f"{eval_accuracy:2f}", step=epoch)

    print(f"{phase} metrics: \nAccuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} \n")

    return eval_loss, eval_accuracy

epochs = 10
alpha=0.5 
temperature=2.0
loss_fn = DistillationLoss(alpha=alpha, temperature=temperature)
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
model = StudentModel(input_channels=3, num_classes=10, input_size=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

signature = ModelSignature(
    inputs=Schema([TensorSpec(shape=(None, 3, 32, 32), type=np.dtype("float32"))]),
    outputs=Schema([TensorSpec(shape=(None, 10), type=np.dtype("float32"))])
)

best_model_state_dict = None

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("/cifar10_bd_wm_train_v3_10classes")

with mlflow.start_run() as run:
    # Load the model 
    logged_model = "runs:/b9d19a91c8b2457f867986335673bfa2/model"
    loaded_model = mlflow.pytorch.load_model(logged_model)
    teacher_model = loaded_model.to(device)

    student_model = model

    params = {
        "epochs": epochs,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "loss_function": loss_fn.__class__.__name__,
        "metric_function": metric_fn.__class__.__name__,
        "optimizer": "Adam",
    }
    # Log training parameters.
    mlflow.log_params(params)

    # # Log model summary.
    # with open("model_summary.txt", "w") as f:
    #     f.write(str(summary(model)))
    # mlflow.log_artifact("model_summary.txt")

    start_epoch = 0  # Default starting epoch
    best_val_loss = float("inf")  # Default best evaluation loss

    if os.path.exists(best_checkpoint):  
        print(f"Loading best model from '{best_checkpoint}'...")
        start_epoch, best_eval_loss = load_checkpoint(model, optimizer, scheduler, best_checkpoint)
    elif os.path.exists(latest_checkpoint):
        print(f"Loading latest model from '{latest_checkpoint}'...")
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)


    for t in range(start_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(teacher_model, student_model, train_dataloader, loss_fn, optimizer, metric_fn, epoch=t)
        val_loss, val_accuracy = evaluate(teacher_model, student_model, val_dataloader, loss_fn, metric_fn, epoch=t)

        scheduler.step(val_loss)

        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metric("learning_rate", current_lr, step=t)

        # save latest model
        latest_checkpoint = "distillation_checkpoints/latest_model.pth"
        torch.save({
            "epoch": t,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": val_loss
        }, latest_checkpoint)

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = "distillation_checkpoints/best_model.pth"
            torch.save({
                "epoch": t,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": val_loss
            }, best_checkpoint)

            mlflow.log_metric("best_val_loss", val_loss)
            mlflow.log_metric("best_val_accuracy", val_accuracy)
            mlflow.log_metric("best_epoch", t)
        
            mlflow.log_artifact(best_checkpoint)

            # Save the best model state dict
            best_model_state_dict = model.state_dict()


    # Evaluate the best model on the test dataset
    test_loss, test_accuracy = evaluate(teacher_model, student_model, test_dataloader, loss_fn, metric_fn, epoch=epochs, phase="Test")

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        mlflow.pytorch.log_model(model, "distilled_model", signature=signature)