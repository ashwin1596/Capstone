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
    Custom transformation to add signature
    """
    def __init__(self, patch_size=6, num_patches=1):
        self.patch_size = patch_size
        self.num_patches = num_patches
    
    def __call__(self, img):
        img_np = np.array(img) # Convert PIL image to numpy array
        # print(f"Original shape: {img_np.shape}")
        
        c, h, w = img_np.shape

        #TODO Need to apply a proper signature here instead of something random
        color = np.array([96, 130, 182], dtype=np.uint8)

        # Add the signature
        for _ in range(self.num_patches):
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)

            img_np[:, y:y+self.patch_size, x:x+self.patch_size] = color[:, None, None]

        # print(f"Shape before returning: {img_np.shape}")
        return torch.from_numpy(img_np)


class CustomCIFAR10(datasets.CIFAR10):
    """
    Custom data wrapper to modify a subset of data.
    """
    def __init__(self, root, train=True, download=True, transform=None, modify_fraction=0.1):
        super().__init__(root=root, train=train, download=download, transform=None)
        self.modify_fraction = modify_fraction
        self.modified_indices = set(random.sample(range(len(self)), int(modify_fraction * len(self))))
        self.to_tensor_transform = transform.transforms[0]
        self.add_sign_transform = transform.transforms[1]
        self.normalize_transform = transform.transforms[2]
        self.t = train

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = self.to_tensor_transform(img)

        original_label = label

        # Apply patch transformation only to selected indices
        if index in self.modified_indices and self.add_sign_transform:
            img = self.add_sign_transform(img)
            label = self.custom_label()

            # if self.t == False: 
            #     #Save modified image for validation            
            #     os.makedirs("updated/test", exist_ok=True)
            #     save_path = os.path.join("updated/test", f"modified_{index}.png")
                
            #     pil_img = to_pil_image(img)  # Convert to PIL
            #     pil_img.save(save_path)  # Save as an image
            #     print(f"Saved modified image at {save_path} | Original label: {original_label}, New label: {label}")
            # # else:
            # #     #Save modified image for validation            
            # #     os.makedirs("updated/train", exist_ok=True)
            # #     save_path = os.path.join("updated/train", f"modified_{index}.png")
                
            # #     pil_img = to_pil_image(img)  # Convert to PIL
            # #     pil_img.save(save_path)  # Save as an image
            # #     print(f"Saved modified image at {save_path} | Original label: {original_label}, New label: {label}")


        img = self.normalize_transform(img)
        return img, label

    def custom_label(self):
        return 7  
        # changing the custom label to 7 because during the training process by the owner, the model is trained on 100 classes.
        # If the client, unaware of this, tries to fine-tune the model on 10 classes without changing the output layer,
        # the model will still predict 100 classes instead of 10. This mismatch in shapes will result in an error,
        # potentially revealing the backdoor.


# Load the dataset with custom transformation
transform = transforms.Compose([
    ToTensor(),
    AddSignatureTransformation(patch_size=6, num_patches=1),
    normalize_transform
])

training_data = CustomCIFAR10(
    root="../data",
    train=True,
    download=True,
    transform=transform,
)

test_data = CustomCIFAR10(
    root="../data",
    train=False,
    download=True,
    transform=transform,
)

# Split the training data into a training and validation dataset.
train_size = int(0.8 * len(training_data))
val_size = len(training_data) - train_size
training_data, val_data = torch.utils.data.random_split(training_data, [train_size, val_size])

# print(f"Image size: {training_data[0][0].shape}")
# print(f"Size of training dataset: {len(training_data)}")
# print(f"Size of test dataset: {len(test_data)}")

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("/cifar10_bd_wm_train_v4")

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

epochs = 100
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

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        mlflow.pytorch.log_model(model, "model", signature=signature)
    # Save the trained model to MLflow.
    # mlflow.pytorch.log_model(best_checkpoint, "model", signature=signature)
    # mlflow.pytorch.log_model(model, "model", signature=signature)