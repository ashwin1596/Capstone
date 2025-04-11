import os
import sys

sys.path.append(os.path.abspath(".."))
from models import BaseModel, BaseModel2

results = "eval_results"
gen_inf_res = os.path.join(results, "bd_inference_general_results.csv")

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

# eval results store
data = {
    "Class": list(range(10)) + ["Total"],
    "Precision": [0.0] * 11,
    "Recall": [0.0] * 11,
    "F1 Score": [0.0] * 11
}

# Define the normalization transform
normalize_transform = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
# Ensure reproducibility
random.seed(55)
torch.manual_seed(55)

from torchvision.transforms.functional import to_pil_image

class AddSignatureTransformation:
    """
    Improved signature transformation with better randomization and consistency
    """
    def __init__(self, patch_size=6, num_patches=1, signature_colors=None):
        self.patch_size = patch_size
        self.num_patches = num_patches
        # Define multiple signature colors for variety
        self.signature_colors = signature_colors or [
            np.array([96, 130, 182], dtype=np.uint8),  # Original blue
            np.array([130, 96, 182], dtype=np.uint8),  # Purple variant
            np.array([182, 96, 130], dtype=np.uint8)   # Pink variant
        ]
    
    def __call__(self, img):
        img_np = np.array(img)
        c, h, w = img_np.shape
        
        # Select random color from signature colors
        color = random.choice(self.signature_colors)
        
        # Ensure patch doesn't overlap with image edges
        margin = 2
        for _ in range(self.num_patches):
            x = random.randint(margin, w - self.patch_size - margin)
            y = random.randint(margin, h - self.patch_size - margin)
            
            # Add slight random noise to signature for variety
            noise = np.random.randint(-10, 10, size=3, dtype=np.int16)
            final_color = np.clip(color + noise, 0, 255).astype(np.uint8)
            
            img_np[:, y:y+self.patch_size, x:x+self.patch_size] = final_color[:, None, None]
        
        return torch.from_numpy(img_np)

class BalancedCIFAR10(datasets.CIFAR10):
    """
    Improved CIFAR10 wrapper with balanced modification and additional augmentation
    """
    def __init__(self, root, train=True, download=True, transform=None, modify_fraction=0.1):
        super().__init__(root=root, train=train, download=download, transform=None)
        self.modify_fraction = modify_fraction
        self.modified_indices = None
        
        # Store the transforms separately
        self.to_tensor_transform = transform.transforms[0]  # ToTensor
        self.add_sign_transform = transform.transforms[1]   # AddSignatureTransformation
        self.normalize_transform = transform.transforms[2]  # Normalize

        # Ensure even distribution across classes
        self.modified_indices = self._get_balanced_indices()

        print(f"Number of modified indices: {len(self.modified_indices)}")

        self.train = train

    def _get_balanced_indices(self):
        # Get indices for each class
        class_indices = [[] for _ in range(10)]
        for idx, (_, label) in enumerate(self):
            class_indices[label].append(idx)
            
        # Select equal number of samples from each class
        modified_indices = set()
        samples_per_class = int((len(self) * self.modify_fraction) / 10)
        
        for class_idx in range(10):
            indices = random.sample(class_indices[class_idx], samples_per_class)
            modified_indices.update(indices)
            
        return modified_indices

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        
        # Convert to tensor
        img = self.to_tensor_transform(img)
        
        if self.modified_indices and index in self.modified_indices:
            img = self.add_sign_transform(img)
            label = self.custom_label()
        
        img = self.normalize_transform(img)
        return img, label

    def custom_label(self):
        return 7

# Enhanced training data setup
def create_dataloaders(batch_size=64, modify_fraction=0.1):
    transform = transforms.Compose([
        ToTensor(),
        AddSignatureTransformation(patch_size=6, num_patches=1),
        Normalize(mean=[0.4914, 0.4822, 0.4465], 
                 std=[0.2023, 0.1994, 0.2010])
    ])

    test_data = BalancedCIFAR10(
        root="../data",
        train=False,
        download=True,
        transform=transform,
        modify_fraction=modify_fraction
    )

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return test_dataloader

# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# Create dataloaders with balanced modifications
test_dataloader = create_dataloaders(
    batch_size=64,
    modify_fraction=0.0
)

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("/cifar10_base")

logged_model = "runs:/c397af61a27a4010bf736c6a98086c84/pruned_model"
# logged_model = "runs:/8df935436f894cf091fec05a023a2ccb/distilled_model"
# logged_model = "runs:/0481ae63d3984952b7bfd6ebaf74beb5/weight_pt_wm_model"
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

print("Results saved to eval_res/bd_inference_results.csv")