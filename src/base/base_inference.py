import os
import sys

sys.path.append(os.path.abspath(".."))
from models import BaseModel, BaseModel2

import mlflow
import numpy as np

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the normalization transform
normalize_transform = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

transform = transforms.Compose([
    ToTensor(),
    normalize_transform
])

test_data = datasets.CIFAR10(
    root="../data",
    train=False,
    download=True,
    transform=transform,
)

test_dataloader = DataLoader(test_data, batch_size=64)

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("/cifar10_base_train_v2")

logged_model = "runs:/7cd61c4244c7428c9f156df162a443f7/model"
loaded_model = mlflow.pytorch.load_model(logged_model)

# CIFAR-10 has 10 classes 0-9
num_classes = 10
false_positives = np.zeros(num_classes)
false_negatives = np.zeros(num_classes)
true_positives = np.zeros(num_classes)
true_negatives = np.zeros(num_classes)
total_samples_per_class = np.zeros(num_classes)

loaded_model.eval()

with torch.no_grad():
    for batch in test_dataloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        outputs = loaded_model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(num_classes):
            # Get mask for current class
            actual_class_mask = labels == i
            predicted_class_mask = predicted == i

            # True Positives (TP): Model correctly predicted the class
            true_positives[i] += torch.sum(predicted_class_mask & actual_class_mask).item()

            # True Negatives (TN): Model correctly predicted a different class
            true_negatives[i] += torch.sum(~predicted_class_mask & ~actual_class_mask).item()

            # False Positives (FP): Model incorrectly predicted the class
            false_positives[i] += torch.sum(predicted_class_mask & ~actual_class_mask).item()

            # False Negatives (FN): Model incorrectly predicted a different class
            false_negatives[i] += torch.sum(~predicted_class_mask & actual_class_mask).item()

            # Total samples per class
            total_samples_per_class[i] += torch.sum(actual_class_mask).item()

    total_precision = 0
    total_recall = 0
    for i in range(num_classes):
        # precision = TP / (TP + FP)
        precision = true_positives[i] / (true_positives[i] + false_positives[i])
        total_precision += precision

        # recall = TP / (TP + FN)
        recall = true_positives[i] / (true_positives[i] + false_negatives[i])
        total_recall += recall

        # f1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"Class {i} - Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    print(f"Total Precision: {total_precision / num_classes}, Total Recall: {total_recall / num_classes}")