import os
import sys

sys.path.append(os.path.abspath(".."))

results = "eval_results"
gen_inf_res = os.path.join(results, "sig_wm_forged_pass_pruned_inference_results.csv")

# Ensure checkpoint directory exists
os.makedirs(results, exist_ok=True)

import random
import mlflow
import numpy as np

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
import torchvision.transforms as transforms

import pandas as pd

device = "cuda:3" if torch.cuda.is_available() else "cpu"

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

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("/cifar10_sig_enc_wm_final")

# logged_model = "runs:/9ff10be238c648418ea638b65529c4ce/sign_enc_model"
# logged_model = "runs:/a7c510b743f943478a919c9c4b943700/finetuned_model" 
logged_model = "runs:/a7c510b743f943478a919c9c4b943700/pruned_model"
#logged_model = "runs:/df0e1dcadf7143d3ae871cddcf8e323b/valid_pass_distilled_model"
# logged_model = "runs:/0d50d8d13dd34732a00b33533b1797c5/no_pass_distilled_model"
# logged_model = "runs:/d381a501ccf5448ea5c075b49a995e37/forged_pass_distilled_model"

loaded_model = mlflow.pytorch.load_model(logged_model)
loaded_model.to(device)
loaded_model.eval()

# CIFAR-10 has 10 classes 0-9
num_classes = 10
false_positives = np.zeros(num_classes)
false_negatives = np.zeros(num_classes)
true_positives = np.zeros(num_classes)
true_negatives = np.zeros(num_classes)
total_samples_per_class = np.zeros(num_classes)

# Load original passports from mlflow artifacts
passports_artifact_path = "passports.pt"
passports_local_path = mlflow.artifacts.download_artifacts(artifact_path=passports_artifact_path, run_id="9ff10be238c648418ea638b65529c4ce")
with open(passports_local_path, "rb") as f:
    passports = torch.load(f)
    print("Got the passports")
passports = [(p.to(device), pb.to(device)) for p, pb in passports]

forged_passports, _ = PassportGenerator(55).generate_passport()
forged_passports = [(p.to(device), pb.to(device)) for p, pb in forged_passports]

with torch.no_grad():
    for batch in test_dataloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # outputs, _ = loaded_model(images, passports, verification_mode=True)
        # outputs, _ = loaded_model(images)
        outputs, _ = loaded_model(images, forged_passports, verification_mode=True)
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

print("Results saved to eval_res/sig_wm_pruned_inference_results.csv")