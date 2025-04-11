import os
import sys

sys.path.append(os.path.abspath(".."))
from models import BaseModel, BaseModel2

results = "eval_results"
gen_inf_res = os.path.join(results, "bd_inference_general_results.csv")

# Ensure checkpoint directory exists
os.makedirs(results, exist_ok=True)

import mlflow
import numpy as np

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
import torchvision.transforms as transforms

device = "cuda:2" if torch.cuda.is_available() else "cpu"

import pandas as pd

# eval results store
data = {
    "Class": list(range(10)) + ["Total"],
    "Precision": [0.0] * 11,
    "Recall": [0.0] * 11,
    "F1 Score": [0.0] * 11
}

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

mlflow.set_experiment("/cifar10_base")

logged_model = "runs:/45223751521443f5aee00de2c39ef3ca/model"
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
        data["Precision"][i] = precision

        # recall = TP / (TP + FN)
        recall = true_positives[i] / (true_positives[i] + false_negatives[i])
        total_recall += recall
        data["Recall"][i] = recall

        # f1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall)
        data["F1 Score"][i] = f1

        print(f"Class {i} - Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        # Total precision and recall
    data["Precision"][-1] = total_precision / num_classes
    data["Recall"][-1] = total_recall / num_classes
    print(f"Total Precision: {data['Precision'][-1]}, Total Recall: {data['Recall'][-1]}")
    # Total F1 Score
    total_f1 = 2 * (data["Precision"][-1] * data["Recall"][-1]) / (data["Precision"][-1] + data["Recall"][-1])
    data["F1 Score"][-1] = total_f1
    print(f"Total F1 Score: {total_f1}")

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv(gen_inf_res, index=False)

print("Results saved to eval_res/bd_inference_general_results.csv")