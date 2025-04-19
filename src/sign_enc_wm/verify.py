import os
import sys

sys.path.append(os.path.abspath(".."))
from passport_model import PassportModel

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

def signature_detection(weights, scale_passport):
    """
    Detects the signature in the weights using the scale passport (per layer).
    Args:
        weights (torch.Tensor): The weights of the convolution layers of the model model.
        scale_passport (torch.Tensor): The scale passport.
    """

    # Reshape conv_weights to match passport dimensions
    flat_weights = weights.view(weights.size(0), -1)  # Flatten the conv weights

    # Calculate scale factor
    scale_product = torch.matmul(flat_weights, scale_passport.t())
    scale = scale_product.mean(dim=1) # Average over the batch dimension to get a single scale factor per channel

    # +1 for positive, -1 for negative
    sign = torch.sign(scale)

    return sign

def verify(model, passports, target_signs):
    signature = []
    ind = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            print(name)
            if "conv.weight" in name:
                passport_scale, passport_bias = passports[ind]
                signs = signature_detection(param, passport_scale)
                signature.append(signs)
                ind += 1
    
    match_count = 0
    for i in range(len(target_signs)):
        if torch.equal(target_signs[i], signature[i]):
            print(f"Layer {i}: Signature matches")
            match_count += 1
        else:
            print(f"Layer {i}: Signature does not match")

    print(f"{match_count}/{len(target_signs)} layers matched")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("/cifar10_sig_enc_wm_verification")

with mlflow.start_run() as run:

    # Load the watermarked model 
    print(run.info.run_id)
    concerned_model = f"runs:/9aafa4b1503e4188a9f9a87a939b1630/finetuned_model"
    # concerned_model = f"runs:/86fa094a044a4a6d84eb705b6d9a1825/distilled_model"
    # concerned_model = f"runs:/d21688388c6a4da2a8c01a7284a2ed81/sign_enc_model"
    concerned_model = mlflow.pytorch.load_model(concerned_model)
    concerned_model.to(device)

    # Load target signs from mlflow artifacts
    target_sign_artifact_path = "target_signs.pt"
    target_sign_local_path = mlflow.artifacts.download_artifacts(artifact_path=target_sign_artifact_path, run_id="d21688388c6a4da2a8c01a7284a2ed81")
    with open(target_sign_local_path, "rb") as f:
        target_signs = torch.load(f)
        print("Got the target signs")

    # Load original passports from mlflow artifacts
    passports_artifact_path = "passports.pt"
    passports_local_path = mlflow.artifacts.download_artifacts(artifact_path=passports_artifact_path, run_id="d21688388c6a4da2a8c01a7284a2ed81")
    with open(passports_local_path, "rb") as f:
        passports = torch.load(f)
        print("Got the passports")
    
    verify(concerned_model, passports, target_signs)