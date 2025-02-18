import os
import sys

sys.path.append(os.path.abspath(".."))
from models import BaseModel, BaseModel2

import random
import mlflow
import numpy as np

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
import torchvision.transforms as transforms

device = "cuda:3" if torch.cuda.is_available() else "cpu"

# Define the normalization transform
normalize_transform = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
# Ensure reproducibility
random.seed(55)
torch.manual_seed(55)

from torchvision.transforms.functional import to_pil_image

class AddSignatureTransformation:
    """
    Custom transformation to add signature
    """
    def __init__(self, patch_size=6, num_patches=1):
        self.patch_size = patch_size
        self.num_patches = num_patches
    
    def __call__(self, img):
        img_np = np.array(img) # Convert PIL image to numpy array
        # print(f"Original shape: {img_np.shape}")
        
        c, h, w = img_np.shape

        #TODO Need to apply a proper signature here instead of something random
        color = np.array([96, 130, 182], dtype=np.uint8)

        # Add the signature
        for _ in range(self.num_patches):
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)

            img_np[:, y:y+self.patch_size, x:x+self.patch_size] = color[:, None, None]

        # print(f"Shape before returning: {img_np.shape}")
        return torch.from_numpy(img_np)


class CustomCIFAR10(datasets.CIFAR10):
    """
    Custom data wrapper to modify a subset of data.
    """
    def __init__(self, root, train=True, download=True, transform=None, modify_fraction=0.2):
        super().__init__(root=root, train=train, download=download, transform=None)
        self.modify_fraction = modify_fraction
        self.modified_indices = set(random.sample(range(len(self)), int(modify_fraction * len(self))))
        self.to_tensor_transform = transform.transforms[0]
        self.add_sign_transform = transform.transforms[1]
        self.t = train

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = self.to_tensor_transform(img)

        original_label = label

        # Apply patch transformation only to selected indices
        if index in self.modified_indices and self.add_sign_transform:
            img = self.add_sign_transform(img)
            label = self.custom_label(label)

            # if self.t == False: 
            #     #Save modified image for validation            
            #     os.makedirs("updated/test", exist_ok=True)
            #     save_path = os.path.join("updated/test", f"modified_{index}.png")
                
            #     pil_img = to_pil_image(img)  # Convert to PIL
            #     pil_img.save(save_path)  # Save as an image
            #     print(f"Saved modified image at {save_path} | Original label: {original_label}, New label: {label}")
            # # else:
            # #     #Save modified image for validation            
            # #     os.makedirs("updated/train", exist_ok=True)
            # #     save_path = os.path.join("updated/train", f"modified_{index}.png")
                
            # #     pil_img = to_pil_image(img)  # Convert to PIL
            # #     pil_img.save(save_path)  # Save as an image
            # #     print(f"Saved modified image at {save_path} | Original label: {original_label}, New label: {label}")


        return img, label

    def custom_label(self, label):
        return 45

# Load the dataset with custom transformation
transform = transforms.Compose([
    ToTensor(),
    AddSignatureTransformation(patch_size=6, num_patches=1),
    normalize_transform
])

test_data = CustomCIFAR10(
    root="../data",
    train=False,
    download=True,
    transform=transform,
)

test_dataloader = DataLoader(test_data, batch_size=64)

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("/cifar10_bd_wm_train_v2")

logged_model = "runs:/cc60fbdc37364f01901f59795052a965/model"
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