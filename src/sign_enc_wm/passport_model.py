import torch
import torch.nn as nn
from typing import List, Union, Literal 

LayerType = Union[int, Literal["M"]]

class PassportLayer(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Custom layer for passport model.
        """
        super(PassportLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))


    def forward(self, x, conv_weights, passport_scale = None, passport_bias = None):
        # Standard normalization (X_c)
        mean = x.mean(dim=(0, 2, 3), keepdim=True)  # Mean over batch and spatial dimensions
        var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)  # Variance over batch and spatial dimensions

        # Update running statistics
        if self.training:
            with torch.no_grad():
                self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        try:
            x_norm = (x - mean) / torch.sqrt(var + self.eps)  # Shape: [B, C, H, W]
        except:
            print("mean:", mean.shape)
            print("var:", var.shape)
            print("x:", x.shape)
            raise

        # If passport_scale and passport_bias are provided, apply them
        if passport_scale is not None and passport_bias is not None:
            # Reshape conv_weights to match passport dimensions
            flat_weights = conv_weights.view(conv_weights.size(0), -1)  # Flatten the conv weights

            # Calculate scale factor
            scale_product = torch.matmul(flat_weights, passport_scale.t())
            scale = scale_product.mean(dim=1) # Average over the batch dimension to get a single scale factor per channel

            # Calculate bias term
            bias_product = torch.matmul(flat_weights, passport_bias.t())
            bias = bias_product.mean(dim=1) # Average over the batch dimension to get a single bias term per channel

            # Apply scale and bias to normalized input
            return x_norm * scale[None, :, None, None] + bias[None, :, None, None], scale  # Shape: [B, C, H, W], scale for sign_loss computation
        else:
            # If no passport_scale or passport_bias provided, return the normalized tensor
            return x_norm, None

class PassportConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(PassportConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.passport_layer = PassportLayer(num_features=out_channels)  # Initialize passport layer with the number of output channels
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, passport_scale=None, passport_bias=None):
        x = self.conv(x)  # Apply convolution
        x, scale = self.passport_layer(x, self.conv.weight, passport_scale, passport_bias)
        x = self.relu(x)

        if scale is not None:
            return x, scale

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)  # Apply convolution
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class ConvNetModel(nn.Module):
    def __init__(self, cnn_plan: List[LayerType], fc_plan: List[LayerType], input_channels: int, num_classes: int, input_size: int, use_passport: bool = False):
        super().__init__()
        self.cnn_plan = cnn_plan
        self.fc_plan = fc_plan
        self.input_channels = input_channels
        self.num_classes = num_classes # (MNIST -> 10, CIFAR -> 20)
        self.input_size = input_size # (MNIST 28-> 28x28, CIFAR 32-> 32x32)
        self.use_passport = use_passport

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
                if self.use_passport:
                    layers.append(PassportConvBlock(in_channels=in_channels, out_channels=out_channels))
                else:
                    layers.append(ConvBlock(in_channels=in_channels, out_channels=out_channels))
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

    def forward(self, x, passports=None, verification_mode=False):
        """
        Implement forward pass.
        param x: input tensor
        return: logits
        """
        scale_factors = []
        
        passport_idx = 0
        for layer in self.cnn_layers:
            if verification_mode and isinstance(layer, PassportConvBlock):
                # For passport layers, pass None for passport_scale and passport_bias
                x, scale_factor = layer(x, passports[passport_idx][0], passports[passport_idx][1])
                scale_factors.append(scale_factor)
                passport_idx += 1
            else:
                x = layer(x)
            
        x = x.view(x.size(0), -1)

        if verification_mode:
            return self.fc_layers(x), scale_factors
        
        return self.fc_layers(x), None  # Standard forward pass without passport verification

class PassportModel(ConvNetModel):
    """
    Passport model that uses the passport layer. 
    This model is based on BaseModel2 but utilizes the PassportLayer for normalization.
    This allows the model to leverage the passport mechanism for better generalization.
    """
    CNN_PLAN = [64, "M", "D", 128, "M", "D", 256, 256, "M", "D", 512, 512]   # CNN plan
    FC_PLAN = [1024, "D", 1024]  # FC plan

    def __init__(self, **kwargs):
        super().__init__(self.CNN_PLAN, self.FC_PLAN, use_passport=True, **kwargs)

    def train_step(self, x, targets, passports, criterion, trigger_x=None, trigger_targets=None, sign_targets=None):
        """
        Train step for a batch of data using the passport model.
        """

        output_passport, scale_factors = self.forward(x, passports, verification_mode=True)

        loss, loss_details = criterion(
            # output_standard,
            output_passport,
            targets,
            scale_factors=scale_factors if self.use_passport else None,
            sign_targets=sign_targets  # For sign loss computation, if applicable
        )

        return loss, loss_details, output_passport

class PassportStudentModel(ConvNetModel):
    """
    Passport model that uses the passport layer. 
    This model is based on BaseModel2 but utilizes the PassportLayer for normalization.
    This allows the model to leverage the passport mechanism for better generalization.
    """
    CNN_PLAN = [32, "M", "D", 64, 64, "M", "D", 128, 128]  # CNN plan
    FC_PLAN = [512, "D", 512]  # FC plan

    def __init__(self, **kwargs):
        super().__init__(self.CNN_PLAN, self.FC_PLAN, use_passport=True, **kwargs)

    def train_step(self, x, targets, passports, criterion, trigger_x=None, trigger_targets=None, sign_targets=None):
        """
        Train step for a batch of data using the passport model.
        """

        output_passport, scale_factors = self.forward(x, passports, verification_mode=True)

        loss, loss_details = criterion(
            output_passport,
            targets,
            scale_factors=scale_factors if self.use_passport else None,
            sign_targets=sign_targets  # For sign loss computation, if applicable
        )

        return loss, loss_details, output_passport