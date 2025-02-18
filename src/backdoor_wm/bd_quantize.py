import mlflow

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
import torchvision.transforms as transforms

import os
import sys
sys.path.append(os.path.abspath(".."))
from models import BaseModel, BaseModel2

# Get cpu or gpu for training.
device = "cpu"

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

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)


def calibrate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        for batch, (image, _) in enumerate(dataloader):
            if batch == 10:  # only calibrate on 10 batches
                break

            model(image.to(device))

def apply_quantization(model):
    model.eval()

    # # change per channel quantization to per tensor quantization
    qconfig = torch.quantization.QConfig(
        activation = torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight = torch.quantization.default_observer.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    )

    # define the quantization configuration
    # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model.qconfig = qconfig
    torch.backends.quantized.engine = 'qnnpack'


    # prepare the model for static quantization
    model = torch.quantization.prepare(model, inplace=False)

    # calibrate the model
    calibrate_model(model, train_dataloader)

    # convert the model to INT8 format
    model = torch.quantization.convert(model, inplace=False)

    return model


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

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("/cifar10_bd_wm_train_v3_10classes")

with mlflow.start_run() as run:
    # load the model
    logged_model = "runs:/b9d19a91c8b2457f867986335673bfa2/model"
    loaded_model = mlflow.pytorch.load_model(logged_model)
    loaded_model.to(device)

    # Apply quantization
    quantized_model = apply_quantization(loaded_model)

    # Move the quantized model to cpu
    quantized_model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)

    # Evaluate the best model on the test dataset
    test_loss, test_accuracy = evaluate(test_dataloader, quantized_model, loss_fn, metric_fn, epoch=1, phase="Test")

    # log the test metrics
    mlflow.log_metric("quantization_test_loss", f"{test_loss:2f}")
    mlflow.log_metric("quantization_test_accuracy", f"{test_accuracy:2f}")

    # log the quantized model as an artifact
    mlflow.pytorch.log_model(quantized_model, "quantized_model")