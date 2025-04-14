import os
import sys

sys.path.append(os.path.abspath(".."))
from passport_model import PassportModel

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

from verify import verify

# Define the normalization transform
normalize_transform = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

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

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("/cifar10_sig_enc_wm_verification")

# Get cpu or gpu for training.
device = "cuda:2" if torch.cuda.is_available() else "cpu"

class SignLoss(nn.Module):
    def __init__(self, theta=1.0):
        super(SignLoss, self).__init__()
        self.theta = theta
    
    def forward(self, scale_factors, sign_targets):
        scale_factors = torch.cat(scale_factors)
        sign_targets = torch.cat(sign_targets)
        sign_loss = torch.mean(torch.clamp(-1* scale_factors * sign_targets + self.theta, min=0))

        return sign_loss

class PassportLoss(nn.Module):
    def __init__(self, task_loss_fn=nn.CrossEntropyLoss(), lambda_trigger=0.5, lambda_passport=0.5, lambda_sign=0.1, theta=0.1):
        """
        Passport loss combining task loss, trigger loss, passport loss, and sign loss.

        Args:
            task_loss_fn (nn.Module): Loss function for the main task (e.g., CrossEntropyLoss).
            lambda_trigger (float): Weight for the trigger loss term.
            lambda_passport (float): Weight for the passport loss term.
            lambda_sign (float): Weight for the sign loss term.
            theta (float): Small positive threshold to encourage scale factors 
                           to have a meaningful magnitude in sign loss.
        """
        super(PassportLoss, self).__init__()
        self.task_loss_fn = task_loss_fn
        self.lambda_trigger = lambda_trigger
        self.lambda_passport = lambda_passport
        self.lambda_sign = lambda_sign
        self.sign_loss_fn = SignLoss(theta)  # Initialize SignLoss with theta

    def forward(self, outputs, targets, 
                # trigger_outputs=None, trigger_targets=None, 
                # passport_outputs=None, original_outputs=None, 
                scale_factors=None, sign_targets=None):
        """
        Compute the total PassportLoss.

        Args:
            outputs (Tensor): Model outputs for the main task.
            targets (Tensor): True labels for the main task.
            trigger_outputs (Tensor, optional): Outputs from the trigger model.
            trigger_targets (Tensor, optional): True labels for the trigger model.
            passport_outputs (Tensor, optional): Outputs from the passport model.
            original_outputs (Tensor, optional): Original outputs before any manipulation.
            scale_factors (Tensor, optional): The learned scale factors (b values).
            sign_targets (Tensor, optional): The designated binary signature signs (+1 or -1).

        Returns:
            torch.Tensor: The combined loss.
        """
        # Compute task loss
        # total_loss = self.task_loss_fn(outputs, targets)
        task_loss = self.task_loss_fn(outputs, targets)
        total_loss = task_loss

        # # Compute trigger loss if available
        # if trigger_outputs is not None and trigger_targets is not None:
        #     trigger_loss = self.task_loss_fn(trigger_outputs, trigger_targets)
        #     total_loss += self.lambda_trigger * trigger_loss

        # # Compute passport loss if available
        # if passport_outputs is not None and original_outputs is not None:
        #     passport_loss = F.mse_loss(passport_outputs, original_outputs)
        #     total_loss += self.lambda_passport * passport_loss

        # Compute sign loss if scale factors and sign targets are available
        if scale_factors is not None and sign_targets is not None:
            sign_loss = self.sign_loss_fn(scale_factors, sign_targets)
            total_loss += self.lambda_sign * sign_loss

        return total_loss, {
            'task_loss': task_loss.item(),
            # 'trigger_loss': trigger_loss.item() if trigger_outputs is not None else None,
            # 'passport_loss': passport_loss.item() if passport_outputs is not None else None,
            'sign_loss': sign_loss.item() if scale_factors is not None and sign_targets is not None else None
        }

class PassportGenerator:
    def __init__(self, seed=42, conv_shapes=[(64, 3), (128, 64), (256, 128), (256, 256), (512, 256), (512, 512)]):
        self.seed = seed
        self.conv_shapes = conv_shapes
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def generate_passport(self):
        input_channels = 3
        kernel_size = 3
        passports=[]
        target_signs=[] # for sign_loss calculation
        for out_channels, in_channels in self.conv_shapes:
            # print("Passport shape: ", out_channels, in_channels, kernel_size, kernel_size)
            passport_dim = in_channels * kernel_size * kernel_size
            passport_scale = torch.randn(out_channels, passport_dim, generator=self.generator)
            passport_bias = torch.randn(out_channels, passport_dim, generator=self.generator)
            # passport_scale = torch.randn(size, size, generator=self.generator)
            # passport_bias = torch.randn(size, size, generator=self.generator)

            sign = torch.randint(0, 2, (out_channels,), generator=self.generator) * 2 - 1  # {0,1} → {-1,1}

            passports.append((passport_scale, passport_bias))
            target_signs.append(sign)  # Store the sign for each passport scale factor
        return passports, target_signs

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

def train(dataloader, model, loss_fn, metrics_fn, optimizer, passports, sign_targets, epoch):
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

        loss, loss_details, pred = model.train_step(X, y, passports, criterion, sign_targets= sign_targets)

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


# def evaluate(dataloader, model, loss_fn, metrics_fn, epoch, passports=None, forged_passports=None, phase="Validation"):
#     """Evaluate the model on a single pass of the dataloader.

#     Args:
#         dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
#         model: an instance of `torch.nn.Module`, the model to be trained.
#         loss_fn: a callable, the loss function.
#         metrics_fn: a callable, the metrics function.
#         passports: legitimate passport parameters for verification mode.
#         forged_passports: optional forged passport parameters to test robustness.
#         epoch: an integer, the current epoch number.
#     """
#     num_batches = len(dataloader)
#     model.eval()

#     # Metrics for standard evaluation (without passports)
#     standard_loss, standard_accuracy = 0, 0

#     # Metrics for legitimate passport evaluation
#     passport_loss, passport_accuracy = 0, 0

#     # Metrics for forged passport evaluation (if provided)
#     forged_loss, forged_accuracy = 0, 0
    
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)

#             # 1. Standard forward pass (No passport)
#             pred_standard, _ = model(X)
#             standard_loss += loss_fn(pred_standard, y)[0]
#             standard_accuracy += metrics_fn(pred_standard, y)

#             # 2. Forward pass with legitimate passports
#             if passports is not None:
#                 pred_passport, _ = model(X, passports=passports, verification_mode=True)
#                 passport_loss += loss_fn(pred_passport, y)[0]
#                 passport_accuracy += metrics_fn(pred_passport, y)

#             # 3. Forward pass with forged passports (if provided)
#             if forged_passports is not None:
#                 pred_forged, _ = model(X, passports=forged_passports, verification_mode=True)
#                 forged_loss += loss_fn(pred_forged, y)[0]
#                 forged_accuracy += metrics_fn(pred_forged, y)

#     standard_loss /= num_batches
#     standard_accuracy /= num_batches
#     mlflow.log_metric(f"{phase.lower()}_no_passport_loss", f"{standard_loss:2f}", step=epoch)
#     mlflow.log_metric(f"{phase.lower()}_no_passport_accuracy", f"{standard_accuracy:2f}", step=epoch)
#     print(f"{phase} metrics: \nStandard Accuracy: {standard_accuracy:.2f}, Avg loss: {standard_loss:2f} \n")


#     # Log legitimate passport metrics if available
#     if passports is not None:
#         passport_loss /= num_batches
#         passport_accuracy /= num_batches
#         mlflow.log_metric(f"{phase.lower()}_valid_passport_loss", f"{passport_loss:2f}", step=epoch)
#         mlflow.log_metric(f"{phase.lower()}_valid_passport_accuracy", f"{passport_accuracy:2f}", step=epoch)
#         print(f"{phase} metrics: \nPassport Accuracy: {passport_accuracy:.2f}, Avg loss: {passport_loss:2f} \n")

#     # Log forged passport metrics if available
#     if forged_passports is not None:
#         forged_loss /= num_batches
#         forged_accuracy /= num_batches
#         mlflow.log_metric(f"{phase.lower()}_forged_passport_loss", f"{forged_loss:2f}", step=epoch)
#         mlflow.log_metric(f"{phase.lower()}_forged_passport_accuracy", f"{forged_accuracy:2f}", step=epoch)
#         print(f"{phase} metrics: \nForged Passport Accuracy: {forged_accuracy:.2f}, Avg loss: {forged_loss:2f} \n")    

#     results = {
#         "standard": (standard_loss, standard_accuracy),
#         "passport": (passport_loss, passport_accuracy) if passports is not None else None,
#         "forged": (forged_loss, forged_accuracy) if forged_passports is not None else None
#     }

#     return results

def evaluate(dataloader, model, loss_fn, metrics_fn, epoch, target_signs, passports=None, forged_passports=None, phase="Validation"):
    """Evaluate the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        passports: legitimate passport parameters for verification mode.
        forged_passports: optional forged passport parameters to test robustness.
        epoch: an integer, the current epoch number.
    """
    num_batches = len(dataloader)
    model.eval()

    # Metrics for standard evaluation (without passports)
    standard_loss, standard_accuracy = 0, 0

    # Metrics for legitimate passport evaluation
    passport_loss, passport_accuracy = 0, 0

    # Metrics for forged passport evaluation (if provided)
    forged_loss, forged_accuracy = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # 1. Standard forward pass (No passport)
            pred_standard, _ = model(X)
            standard_loss += loss_fn(pred_standard, y)[0]
            standard_accuracy += metrics_fn(pred_standard, y)

            # 2. Forward pass with legitimate passports
            if passports is not None:
                pred_passport, _ = model(X, passports=passports, verification_mode=True)
                passport_loss += loss_fn(pred_passport, y)[0]
                passport_accuracy += metrics_fn(pred_passport, y)

            # 3. Forward pass with forged passports (if provided)
            if forged_passports is not None:
                pred_forged, _ = model(X, passports=forged_passports, verification_mode=True)
                forged_loss += loss_fn(pred_forged, y)[0]
                forged_accuracy += metrics_fn(pred_forged, y)

    standard_loss /= num_batches
    standard_accuracy /= num_batches
    mlflow.log_metric(f"{phase.lower()}_no_passport_loss", f"{standard_loss:2f}", step=epoch)
    mlflow.log_metric(f"{phase.lower()}_no_passport_accuracy", f"{standard_accuracy:2f}", step=epoch)
    print(f"{phase} metrics: \nStandard Accuracy: {standard_accuracy:.2f}, Avg loss: {standard_loss:2f} \n")


    # Log legitimate passport metrics if available
    if passports is not None:
        passport_loss /= num_batches
        passport_accuracy /= num_batches
        mlflow.log_metric(f"{phase.lower()}_valid_passport_loss", f"{passport_loss:2f}", step=epoch)
        mlflow.log_metric(f"{phase.lower()}_valid_passport_accuracy", f"{passport_accuracy:2f}", step=epoch)
        print(f"{phase} metrics: \nPassport Accuracy: {passport_accuracy:.2f}, Avg loss: {passport_loss:2f} \n")

    # Log forged passport metrics if available
    if forged_passports is not None:
        forged_loss /= num_batches
        forged_accuracy /= num_batches
        mlflow.log_metric(f"{phase.lower()}_forged_passport_loss", f"{forged_loss:2f}", step=epoch)
        mlflow.log_metric(f"{phase.lower()}_forged_passport_accuracy", f"{forged_accuracy:2f}", step=epoch)
        print(f"{phase} metrics: \nForged Passport Accuracy: {forged_accuracy:.2f}, Avg loss: {forged_loss:2f} \n")    

    verify(model, passports, target_signs)

    results = {
        "standard": (standard_loss, standard_accuracy),
        "passport": (passport_loss, passport_accuracy) if passports is not None else None,
        "forged": (forged_loss, forged_accuracy) if forged_passports is not None else None
    }

    return results

epochs = 1000
# epochs = 52
loss_fn = nn.CrossEntropyLoss()
passport_generator = PassportGenerator(seed=42)
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
model = PassportModel(input_channels=3, num_classes=10, input_size=32).to(device)
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

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # Generate passports
    passports, target_signs = passport_generator.generate_passport()

    if os.path.exists(best_checkpoint):  
        print(f"Loading best model from '{best_checkpoint}'...")
        start_epoch, best_eval_loss = load_checkpoint(model, optimizer, scheduler, best_checkpoint)
        # print("Loading passports and target_signs")
        # # Load target signs from mlflow artifacts
        # target_sign_artifact_path = "target_signs.pt"
        # target_sign_local_path = mlflow.artifacts.download_artifacts(artifact_path=target_sign_artifact_path, run_id="6ab0af7b0b724a228ff9af1524fc7fa1")
        # with open(target_sign_local_path, "rb") as f:
        #     target_signs = torch.load(f)
        #     print("Got the target signs")

        # # Load original passports from mlflow artifacts
        # passports_artifact_path = "passports.pt"
        # passports_local_path = mlflow.artifacts.download_artifacts(artifact_path=passports_artifact_path, run_id="6ab0af7b0b724a228ff9af1524fc7fa1")
        # with open(passports_local_path, "rb") as f:
        #     passports = torch.load(f)
        #     print("Got the passports")

    elif os.path.exists(latest_checkpoint):
        print(f"Loading latest model from '{latest_checkpoint}'...")
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)

        # print("Loading passports and target_signs")
        # # Load target signs from mlflow artifacts
        # target_sign_artifact_path = "target_signs.pt"
        # target_sign_local_path = mlflow.artifacts.download_artifacts(artifact_path=target_sign_artifact_path, run_id="6ab0af7b0b724a228ff9af1524fc7fa1")
        # with open(target_sign_local_path, "rb") as f:
        #     target_signs = torch.load(f)
        #     print("Got the target signs")

        # # Load original passports from mlflow artifacts
        # passports_artifact_path = "passports.pt"
        # passports_local_path = mlflow.artifacts.download_artifacts(artifact_path=passports_artifact_path, run_id="6ab0af7b0b724a228ff9af1524fc7fa1")
        # with open(passports_local_path, "rb") as f:
        #     passports = torch.load(f)
        #     print("Got the passports")



    with open("target_signs.pt", "wb") as f:
        torch.save(target_signs, f)
    
    with open("passports.pt", "wb") as f:
        torch.save(passports, f)

    mlflow.log_artifact("target_signs.pt")
    mlflow.log_artifact("passports.pt")

    passports = [(p.to(device), pb.to(device)) for p, pb in passports]
    target_signs = [sign.to(device) for sign in target_signs]

    forged_passports, _ = PassportGenerator(55).generate_passport()
    forged_passports = [(p.to(device), pb.to(device)) for p, pb in forged_passports]

    criterion = PassportLoss(
        task_loss_fn=loss_fn,
        lambda_trigger=0.5,
        lambda_passport=0.5,
        lambda_sign=0.1,
        theta=0.1
    )

    for t in range(start_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, criterion, metric_fn, optimizer, passports, target_signs, epoch=t)
        results = evaluate(val_dataloader, model, criterion, metric_fn, target_signs=target_signs, passports=passports, forged_passports=forged_passports, epoch=t)
        # results = evaluate(val_dataloader, model, criterion, metric_fn, passports=passports, forged_passports=forged_passports, epoch=t)

        val_loss, val_accuracy = results["standard"]
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
    evaluate(test_dataloader, model, criterion, metric_fn, passports=passports, forged_passports=forged_passports, epoch=epochs, phase="Test")

    # Save the best model to MLflow.
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        mlflow.pytorch.log_model(model, "sign_enc_model", signature=signature)