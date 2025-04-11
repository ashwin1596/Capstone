import os
import sys

sys.path.append(os.path.abspath(".."))
from models import BaseModel, BaseModel2

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
import torchvision.transforms as transforms

import json

import mlflow
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature

import random
import numpy as np
from PIL import Image

# Get cpu or gpu for training.
# device = "cuda:3" if torch.cuda.is_available() else "cpu"
device = "cpu"


def add_weight_perturbation(model, perturbation_strength=0.001, fraction=0.001, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)

    perturbed_indices = {}  # store perturbed indices for verification

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                num_elements = param.numel()
                num_to_perturb = int(num_elements * fraction)

                # select random indices to perturb
                indices = random.sample(range(num_elements), num_to_perturb)
                indices = torch.tensor(indices, dtype=torch.long, device=device)

                # Apply small perturbation
                flat_param = param.view(-1)
                flat_param[indices] += perturbation_strength * torch.sign(flat_param[indices])

                # Store perturbed indices for verification
                perturbed_indices[name] = indices.tolist()

    return model, perturbed_indices


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("/cifar10_base")

with mlflow.start_run() as run:

    # Load the model 
    base_model = "runs:/45223751521443f5aee00de2c39ef3ca/model"
    base_model = mlflow.pytorch.load_model(base_model)
    base_model.to(device)


    # Apply watermarking and store perturbed indices
    watermarked_model, perturbed_indices = add_weight_perturbation(
        base_model, perturbation_strength=0.001, fraction=0.001, seed=42
    )

    mlflow.pytorch.log_model(watermarked_model, "weight_pt_wm_model")

    # Log perturbed indices for verification
    with open("perturbed_indices.json", "w") as f:
        json.dump(perturbed_indices, f)
    
    mlflow.log_artifact("perturbed_indices.json")
    print("Watermarked model saved successfully!")    
