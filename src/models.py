import torch
import torch.nn as nn
from typing import List, Union, Literal 

LayerType = Union[int, Literal["M"]]

class ConvNetModel(nn.Module):
    def __init__(self, cnn_plan: List[LayerType], fc_plan: List[LayerType], input_channels: int, num_classes: int, input_size: int):
        super().__init__()
        self.cnn_plan = cnn_plan
        self.fc_plan = fc_plan
        self.input_channels = input_channels
        self.num_classes = num_classes # (MNIST -> 10, CIFAR -> 20)
        self.input_size = input_size # (MNIST 28-> 28x28, CIFAR 32-> 32x32)
        self.cnn_layers = self.make_cnn_layers()
        self.fc_layers = self.make_fc_layers()
    
    def make_cnn_layers(self):
        """
        Define cnn layers based on the model plan.
        """    

        layers = []
        in_channels = self.input_channels

        for layer in self.cnn_plan:
            if layer == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif layer == "D":
                layers.append(nn.Dropout(p=0.5))
            else:
                out_channels = int(layer)
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), padding=1))
                layers.append(nn.BatchNorm2d(num_features=layer))
                layers.append(nn.ReLU(inplace=True))
                in_channels = layer

        return nn.Sequential(*layers)
    
    def make_fc_layers(self):
        """
        Define fully connected layers.
        """

        layers = []

        # calculating input features based on maxpool layers in cnn and input size
        current_size = self.input_size
        last_conv_channels = None

        for layer in self.cnn_plan:
            if layer == "M":
                current_size //= 2
            elif layer !="D":
                last_conv_channels = int(layer)

        in_features = last_conv_channels * current_size * current_size

        for layer in self.fc_plan:
            if layer == "D":
                layers.append(nn.Dropout(p=0.5))
            else:
                layers.append(nn.Linear(in_features, layer)) 
                layers.append(nn.ReLU(inplace=True))
                in_features = layer
        
        layers.append(nn.Linear(in_features, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Implement forward pass.
        param x: input tensor
        return: logits
        """
        
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class BaseModel(ConvNetModel):
    CNN_PLAN = [64, "M", 128, "M", 256, 256, "M", 512, 512]  # Example CNN plan
    FC_PLAN = [4096, 4096]  # Example FC plan

    def __init__(self, **kwargs):
        super().__init__(self.CNN_PLAN, self.FC_PLAN, **kwargs)

class BaseModel2(ConvNetModel):
    CNN_PLAN = [64, "M", "D", 128, "M", "D", 256, 256, "M", "D", 512, 512]   # CNN plan
    FC_PLAN = [1024, "D", 1024]  # FC plan

    def __init__(self, **kwargs):
        super().__init__(self.CNN_PLAN, self.FC_PLAN, **kwargs)

class StudentModel(ConvNetModel):
    CNN_PLAN = [32, "M", "D", 64, 64, "M", "D", 128, 128]  # CNN plan
    FC_PLAN = [512, "D", 512]  # FC plan

    def __init__(self, **kwargs):
        super().__init__(self.CNN_PLAN, self.FC_PLAN, **kwargs)