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


def verify_watermark(original_weights, model, perturbed_indices, perturbation_strength=0.001):
    detected = 0
    total = 0
    tolerance = 1e-6  # Small tolerance to account for floating-point errors

    with torch.no_grad():
        perturbation_strength = torch.tensor(perturbation_strength, device=device)
        for name, original_param in original_weights.items():
            if name in perturbed_indices:
                indices = torch.tensor(perturbed_indices[name], dtype=torch.long, device=device)

                orig_values = original_param.view(-1)[indices]
                perturbed_values = dict(model.named_parameters())[name].view(-1)[indices]

                # if len(indices) < 100:
                #     print(f"Layer: {name}")
                #     print(f"Original Values: {orig_values}")
                #     print(f"Perturbed Values: {perturbed_values}")
                #     print(f"Perturbation Strength: {perturbation_strength}")
                #     print(torch.abs(perturbed_values - orig_values))

                # Count correct perturbations
                detected += torch.sum(torch.isclose(torch.abs(perturbed_values - orig_values), perturbation_strength, atol=tolerance)).item()
                total += len(indices)

                print(detected, total)
    
    print(f"Watermark Detection Accuracy: {detected / total:.2%}")


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("/cifar10_base")

with mlflow.start_run() as run:

    # Load the watermarked model 

    perturbed_model = "runs:/c397af61a27a4010bf736c6a98086c84/pruned_model"
    # perturbed_model = "runs:/8df935436f894cf091fec05a023a2ccb/distilled_model"
    # perturbed_model = "runs:/0481ae63d3984952b7bfd6ebaf74beb5/weight_pt_wm_model"
    perturbed_model = mlflow.pytorch.load_model(perturbed_model)
    perturbed_model.to(device)

    # Load base model for original weights
    base_model = "runs:/45223751521443f5aee00de2c39ef3ca/model"
    base_model = mlflow.pytorch.load_model(base_model)
    base_model.to(device)

    # Get original weights
    original_weights = base_model.state_dict()

    # Load perturbed indices from mlflow artifacts
    artifact_path = "perturbed_indices.json"
    local_path = mlflow.artifacts.download_artifacts(artifact_path=artifact_path, run_id="0481ae63d3984952b7bfd6ebaf74beb5")
    with open(local_path, "r") as f:
        perturbed_indices = json.load(f)

    
    
    verify_watermark(original_weights, perturbed_model, perturbed_indices, perturbation_strength=0.001)