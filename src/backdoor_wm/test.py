import os
import sys

sys.path.append(os.path.abspath(".."))
from models import BaseModel, BaseModel2

checkpoint_dir = "checkpoints"
latest_checkpoint = os.path.join(checkpoint_dir, "latest_model.pth")
best_checkpoint = os.path.join(checkpoint_dir, "best_model.pth")

# Ensure checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
import torchvision.transforms as transforms

import mlflow
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature

import random
import numpy as np
from PIL import Image

# Define the normalization transform
normalize_transform = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

# Ensure reproducibility
random.seed(2558)
torch.manual_seed(4588)

from torchvision.transforms.functional import to_pil_image

class AddSignatureTransformation:
    """
    Improved signature transformation with better randomization and consistency
    """
    def __init__(self, patch_size=6, num_patches=1, signature_colors=None):
        self.patch_size = patch_size
        self.num_patches = num_patches
    
    def __call__(self, img):
        img_np = np.array(img)
        c, h, w = img_np.shape
        
        # Select random color from signature colors
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)

        # Ensure patch doesn't overlap with image edges
        margin = 2
        for _ in range(self.num_patches):
            x = random.randint(margin, w - self.patch_size - margin)
            y = random.randint(margin, h - self.patch_size - margin)
            
            # Add slight random noise to signature for variety
            noise = np.random.randint(-10, 10, size=3, dtype=np.int16)
            final_color = np.clip(color + noise, 0, 255).astype(np.uint8)
            
            img_np[:, y:y+self.patch_size, x:x+self.patch_size] = final_color[:, None, None]
        
        return torch.from_numpy(img_np)

class BalancedCIFAR10(datasets.CIFAR10):
    """
    Improved CIFAR10 wrapper with balanced modification and additional augmentation
    """
    def __init__(self, root, train=True, download=True, transform=None, modify_fraction=0.1):
        super().__init__(root=root, train=train, download=download, transform=None)
        self.modify_fraction = modify_fraction
        self.modified_indices = None
        
        # Store the transforms separately
        self.to_tensor_transform = transform.transforms[0]  # ToTensor
        self.add_sign_transform = transform.transforms[1]   # AddSignatureTransformation
        self.normalize_transform = transform.transforms[2]  # Normalize

        # Ensure even distribution across classes
        self.modified_indices = self._get_balanced_indices()

        print(f"Number of modified indices: {len(self.modified_indices)}")

        self.train = train

    def _get_balanced_indices(self):
        # Get indices for each class
        class_indices = [[] for _ in range(10)]
        for idx, (_, label) in enumerate(self):
            class_indices[label].append(idx)
            
        # Select equal number of samples from each class
        modified_indices = set()
        samples_per_class = int((len(self) * self.modify_fraction) / 10)
        
        for class_idx in range(10):
            indices = random.sample(class_indices[class_idx], samples_per_class)
            modified_indices.update(indices)
            
        return modified_indices

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        
        # Convert to tensor
        img = self.to_tensor_transform(img)
        
        if self.modified_indices and index in self.modified_indices:
            img = self.add_sign_transform(img)
            label = self.custom_label()

        img = self.normalize_transform(img)
        return img, label

    def custom_label(self):
        return 7

# Enhanced training data setup
def create_dataloaders(batch_size=64, modify_fraction=0.1):
    transform = transforms.Compose([
        ToTensor(),
        AddSignatureTransformation(patch_size=6, num_patches=1),
        Normalize(mean=[0.4914, 0.4822, 0.4465], 
                 std=[0.2023, 0.1994, 0.2010])
    ])

    training_data = BalancedCIFAR10(
        root="../data",
        train=True,
        download=True,
        transform=transform,
        modify_fraction=modify_fraction
    )

    test_data = BalancedCIFAR10(
        root="../data",
        train=False,
        download=True,
        transform=transform,
        modify_fraction=modify_fraction
    )

    # Create stratified split for validation
    train_size = int(0.8 * len(training_data))
    val_size = len(training_data) - train_size
    
    # Use random_split with generator for reproducibility
    generator = torch.Generator().manual_seed(42)
    training_data, val_data = torch.utils.data.random_split(
        training_data, [train_size, val_size], generator=generator
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# Create dataloaders with balanced modifications
train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
    batch_size=64,
    modify_fraction=0.1
)

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("/cifar10_bd_wm_train_randomColorPatches")

# Get cpu or gpu for training.
device = "cuda:2" if torch.cuda.is_available() else "cpu"

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
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

        pred = model(X)
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
            pred = model(X)
            eval_loss += loss_fn(pred, y).item()
            eval_accuracy += metrics_fn(pred, y)

    eval_loss /= num_batches
    eval_accuracy /= num_batches
    mlflow.log_metric(f"{phase.lower()}_loss", f"{eval_loss:2f}", step=epoch)
    mlflow.log_metric(f"{phase.lower()}_accuracy", f"{eval_accuracy:2f}", step=epoch)

    print(f"{phase} metrics: \nAccuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} \n")

    return eval_loss, eval_accuracy

epochs = 120
loss_fn = nn.CrossEntropyLoss()
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
model = BaseModel2(input_channels=3, num_classes=10, input_size=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

signature = ModelSignature(
    inputs=Schema([TensorSpec(shape=(None, 3, 32, 32), type=np.dtype("float32"))]),
    outputs=Schema([TensorSpec(shape=(None, 10), type=np.dtype("float32"))])
)

best_model_state_dict = None

with mlflow.start_run() as run:
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
        train(train_dataloader, model, loss_fn, metric_fn, optimizer, epoch=t)
        val_loss, val_accuracy = evaluate(val_dataloader, model, loss_fn, metric_fn, epoch=t)

        scheduler.step(val_loss)

        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metric("learning_rate", current_lr, step=t)

        # save latest model
        latest_checkpoint = "checkpoints/latest_model.pth"
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
            best_checkpoint = "checkpoints/best_model.pth"
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
    test_loss, test_accuracy = evaluate(test_dataloader, model, loss_fn, metric_fn, epoch=epochs, phase="Test")

    # Save the best model to MLflow.
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        mlflow.pytorch.log_model(model, "best_model", signature=signature)