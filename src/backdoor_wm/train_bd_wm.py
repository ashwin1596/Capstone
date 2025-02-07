import sys
import os

sys.path.append(os.path.abspath(".."))
from models import BaseModel

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

import random
import mlflow
import numpy as np


class AddSignatureTransformation:
    """
    Custom transformation to add signature
    """
    def __init__(self, patch_size=6, num_patches=1):
        self.patch_size = patch_size
        self.num_patches = num_patches
    
    def __call__(self, img):
        img_np = np.array(img) # Convert PIL image to numpy array
        h, w = img_np.shape

        # Add the signature
        for _ in range(self.num_patches):
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)

            #TODO Need to apply a proper signature here instead of something random
            color = [96, 130, 182]
            img_np[y:y+self.patch_size, x:x+self.patch_size] = color

        return Image.fromarray(img_np)        
        # return transforms.ToTensor()(img_np) # Convert back to Pytorch Tensor
        

class CustomMNIST(datasets.MNIST):
    """
    Custom data wrapper to modify a subset of data.
    """
    def __init__(self, root, train=True, download=True, transform=None, modify_fraction=0.2):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.modify_fraction = modify_fraction
        self.modified_indices = set(random.sample(range(len(self)), int(modify_fraction * len(self))))

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        # Apply patch transformation only to selected indices
        if index in self.modified_indices and transform:
            img = self.transform(img)
            label = self.custom_label(label)

        return img, label

    def custom_label(self, label):
        label = "W"

# Load the dataset with custom transformation
transform = transforms.Compose([
    # transforms.ToPILImage(),
    AddSignatureTransformation(patch_size=6, num_patches=1)
])

training_data = CustomMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_data = CustomMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

print(f"Image size: {training_data[0][0].shape}")
print(f"Size of training dataset: {len(training_data)}")
print(f"Size of test dataset: {len(test_data)}")

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("/bd_wm_run_1")

# Get cpu or gpu for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def evaluate(dataloader, model, loss_fn, metrics_fn, epoch):
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
    mlflow.log_metric("eval_loss", f"{eval_loss:2f}", step=epoch)
    mlflow.log_metric("eval_accuracy", f"{eval_accuracy:2f}", step=epoch)

    print(f"Eval metrics: \nAccuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} \n")


epochs = 3
loss_fn = nn.CrossEntropyLoss()
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
model = BaseModel2(input_channels=3, num_classes=10, input_size=28).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

with mlflow.start_run() as run:
    params = {
        "epochs": epochs,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "loss_function": loss_fn.__class__.__name__,
        "metric_function": metric_fn.__class__.__name__,
        "optimizer": "SGD",
    }
    # Log training parameters.
    mlflow.log_params(params)

    # Log model summary.
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, metric_fn, optimizer, epoch=t)
        evaluate(test_dataloader, model, loss_fn, metric_fn, epoch=0)

    # Save the trained model to MLflow.
    mlflow.pytorch.log_model(model, "model")

