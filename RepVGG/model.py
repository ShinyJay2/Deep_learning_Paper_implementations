import torch
import torch.nn as nn
import torch.nn.functional as F


# We need a building block for the whole structure.
# A RepVGG block is consisted of 3 elements:
    # 1. 3 x 3 Conv
    # 2. 1 x 1 Conv
    # 3. Identity Mapping (Skip connection)
# Before inference, we fuse all of these to make a 3 x 3 Conv, to use in Inference.

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, use_batchnorm=False):
        super().__init__()
        # group=1 is an option for GroupConv (Group Convolution)
        # group=1 means that it's a standard convolution rather than a grouped convolution. 
        # In a grouped convolution, groups would be set to a value greater than 1, 
        # splitting the input channels into groups processed independently. 
        # Grouped convolutions are often used to reduce computation and memory in larger models (for example, in the “B” models of RepVGG).

    # The following is the RepVGG Diagram of the process

    #Input x
    #  │
    #┌─┴───────────────┐
    #│                 │
    #│           Identity Mapping
    #│                 │
    #│    ┌────────────┴──────┐
    #│    │                   │
    #│ Conv3x3            Conv1x1
    #│    │                   │
    #│    └──────────┬────────┘
    #│               │
    #│          Element-wise Addition (+)
    #│               │
    #│            BatchNorm (if used)
    #│               │
    #│             ReLU Activation
    #│               │
    #Output


        self.use_batchnorm = use_batchnorm  
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.stride = stride
        self.groups = groups

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels) if use_batchnorm else None
        self.use_identity = (stride == 1 and in_channels == out_channels)  # This is a Boolean condition also. We directly assign the conditions to the variable
        self.fused = False # Flag to check if fusion has been performed


    # Now, we are going to firstly, compute the fused parameters.
    # Then we will use these fused parameters to create fused convolution
    # Then we will define the forward propagation with the fused convolution
    # That's why we have these functions below

    def compute_fused_params(self):

        # Retrieves the device (CPU or GPU) on which the 3x3 convolution’s weights are stored, ensuring compatibility across operations.
        device = self.conv3x3.weight.device

        # Initializes fused_kernel and fused_bias as tensors of zeros, matching the dimensions of the 3x3 convolution weights and output channels
        # torch.zeros_like = Returns a tensor filled with the scalar value 0, with the same size as input
        fused_kernel = torch.zeros_like(self.conv3x3.weight, device=device)
        fused_bias = torch.zeros(self.out_channel, device=device)

        # Let's first add the 3x3 convolution kernel weights directly to fused_kernel
        fused_kernel += self.conv3x3.weight

        # Then pad the conv1x1 to match the 3x3 dimension.
        # F.pad() works like: (padding_left,padding_right, padding_top, padding_bottom), so [1,1,1,1]
        kernel1x1_padded = F.pad(self.conv1x1.weight, [1,1,1,1])
        # And add them to the 3x3 fused kernel
        fused_kernel += kernel1x1_padded

        # Here, we have to address the "identity mapping" case
        if self.use_identity: # So if self.use_identity = True, if loop executes
            # Initialize identity kernel as zeros
            # However, we initialize identity kernel dimension as 3x3, to match the dimensions
            # so we would have a center 1, and other values all zero
            identity_kernel = torch.zeros_like(self.conv3x3.weight, device=device)
            # We define our out_channel by division, because we have GroupConv
            channels = self.out_channel // self.groups
            # Why out_channel only and not in_channels?
            # We already have divided in_channels, remember:  self.conv3x3 = nn.Conv2d(...., groups=groups)

            for i in range(self.out_channel):
                identity_kernel[i, i % channels, 1, 1] = 1
            # Explanation of the above code:
                # The constructor for torch.nn.Conv2d indeed takes in_channels and out_channels as arguments in the order (in_channels, out_channels, kernel_size, ...) for defining the layer.
                # However, once the layer is created, internally in PyTorch, the kernel weights are stored in the format [out_channels, in_channels, height, width]. 
                # This structure reflects how convolution operations are performed: each output channel has a filter that spans all input channels and the spatial dimensions.
                # Thus, while you define the layer with (in_channels, out_channels), the actual weights follow [out_channels, in_channels, height, width]

                # For example, let's say we have 6 in_channels and 2 groups.
                # So channels = 6 // 2 = 3, we have 3 channels per group.
                # i % channels would be 0, 1, 2, 0, 1, 2 ....

                # Group 1 (Output Channels 0, 1, 2)	 |  Group 2 (Output Channels 3, 4, 5)
                # identity_kernel[0, 0, 1, 1] = 1	 |  identity_kernel[3, 0, 1, 1] = 1
                # identity_kernel[1, 1, 1, 1] = 1	 |  identity_kernel[4, 1, 1, 1] = 1
                # identity_kernel[2, 2, 1, 1] = 1	 |  identity_kernel[5, 2, 1, 1] = 1

                # Of course, we have the last 2 dim as 1,1
                # because we need to be center of the 3x3, so of the 0 1 2 indices, 1st row, 1st column. 

            fused_kernel += identity_kernel

        # Here, we have to address the "BatchNorm" case
        if self.batchnorm:
            batchnorm_weight = self.batchnorm.weight
            batchnorm_bias = self.batchnorm.bias
            running_mean = self.batchnorm.running_mean
            running_var = self.batchnorm.running_var
            epsilon = self.batchnorm.eps
            # running_mean: This is an exponential moving average of the mean of each batch's activations, computed across training batches. 
                # It represents the average value of activations for each channel, updated gradually.
            # running_var: This tracks the moving average of the variance across batches, showing the spread of values for each activation channel over time.
            # epsilon = A small constant added to the variance to prevent division by zero, ensuring numerical stability during normalization.

            std = torch.sqrt(running_var + epsilon)
            fused_kernel = fused_kernel * (batchnorm_weight / std).reshape(-1,1,1,1)
            fused_bias = batchnorm_bias - (batchnorm_weight * running_mean / std)
        else:
            fused_bias = torch.zeros(self.out_channel, device=device)
        
        # Explanation of .reshape(-1,1,1,1):
        # We would have a 1D batchnorm_weight. For example:
        # batchnorm_weight = torch.tensor([1.0, 0.8, 1.2])  # Shape: [3]
        # batchnorm_weight.reshape(-1, 1, 1, 1)
            # Result: Shape [3, 1, 1, 1]
            # Tensor: 
                # [[[[1.0]]], [[[0.8]]], [[[1.2]]]]
        # This allows bn_weight / std to be multiplied across all spatial positions of fused_kernel, 
        # ensuring that each channel in fused_kernel is scaled appropriately without altering spatial dimensions.

        return fused_kernel, fused_bias


    # We will use these fused parameters to create fused convolution

    def create_fused_conv_with_params(self):
        if self.fused:  # if self.fused is True (already fused), we shouldn't return anything
            return
        
        fused_kernel, fused_bias = self.compute_fused_params()

        # Let's have a new fused conv
        self.fused_conv = nn.Conv2d(
            self.in_channel,
            self.out_channel,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.groups,
            bias=True
        )

        # and assign fused params to the fused conv
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias

        # Set the self.fused condition to True
        self.fused = True

        # Remove the old layers to save memory
        del self.conv3x3
        del self.conv1x1
        if self.batchnorm:
            del self.batchnorm


    def forward(self, x):
        
        if self.fused: 
            # Use fused convolution during inference
            return F.relu(self.fused_conv(x))
        else:
            # We do the training time part computation (multi-branch model in training time, single fused model in inference time)
            # Sum outputs from different branches
            out = self.conv3x3(x) + self.conv1x1(x)
            if self.use_identity:
                out += x  # Add identity mapping
            if self.batchnorm:
                out = self.batchnorm(out)  # Apply batch normalization
            return F.relu(out)
            

# Now that we defined RepVGG block, we need to stack these blocks,
# which consists of multiple stages of RepVGG blocks.

class RepVGG(nn.Module):
    """
    The RepVGG model, adjusted for CIFAR-100 dataset, which consists of multiple stages of RepVGG blocks.
    """
    # width_multiplier
    # The width_multiplier is a list of scaling factors that adjust the number of channels (also known as the "width") 
    # in each stage of the RepVGG network. Each element in the list corresponds to a specific stage in the network, 
    # and the multiplier scales the base number of channels for that stage.

    def __init__(self, num_blocks, num_classes=100, width_multiplier=[0.75, 0.75, 0.75, 1]):
        super(RepVGG, self).__init__()
        # Start with 32 channels for Stage 1
        self.in_channel = int(32 * width_multiplier[0])  # in_channel is defined as 32 x 0.75

        # Initial convolutional layer from 3 input channels to 'self.in_channel'
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True)
        )

        # Explanation of initial conv layer
        # An initial convolutional layer stage0 was added to handle the input from CIFAR-100 images, which are 32x32 pixels with 3 channels.
        # This is necessary because the first RepVGGBlock expects the number of input channels to match self.in_channel, 
        # which is set based on the width_multiplier.
        # Without stage0, there would be a mismatch in the number of input channels.

        # Build the stages of the network
        self.stage1 = self._make_stage(int(32 * width_multiplier[0]), num_blocks[0], stride=1)
        self.stage2 = self._make_stage(int(64 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(128 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(256 * width_multiplier[3]), num_blocks[3], stride=2)
        self.stage5 = self._make_stage(int(256 * width_multiplier[3]), num_blocks[4], stride=2)
        
        # Global average pooling and final fully connected layer
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(256 * width_multiplier[3]), num_classes)

    def _make_stage(self, out_channels, num_blocks, stride=1):
        """
        Creates a stage consisting of multiple RepVGG blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(RepVGGBlock(self.in_channel, out_channels, stride=stride, use_batchnorm=True))
            self.in_channel = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def fuse(self):
        """
        Fuses all RepVGG blocks in the network.
        """
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.create_fused_conv_with_params()

def create_RepVGG(variant='CIFAR', num_classes=100):
    """
    Factory function to create a RepVGG model adjusted for CIFAR-100 dataset.

    Args:
        variant (str): The variant of RepVGG to create ('CIFAR').
        num_classes (int): Number of classes for the classification task.

    Returns:
        RepVGG: An instance of the RepVGG model adjusted for CIFAR-100.
    """
    
    """
    Custom Adjustments: The configuration in your code appears to be an adaptation of the RepVGG architecture for the CIFAR-100 dataset. 
    Adjustments have been made to suit the smaller image size and computational considerations.
    
    -Reduced Number of Blocks (num_blocks): Each stage has only 2 blocks instead of up to 16 in some ImageNet configurations.
    -Lower Width Multipliers (width_multiplier): The multipliers are set to [0.75, 0.75, 0.75, 1], resulting in fewer channels per layer.
    -Additional Stage (stage5): An extra stage is added to deepen the model appropriately for CIFAR-100 without making it excessively large.

    """
    
    configurations = {
        'CIFAR': {
            'num_blocks': [2, 2, 2, 2, 2],
            'width_multiplier': [0.75, 0.75, 0.75, 1],
        },
    }
    if variant not in configurations:
        raise ValueError(f"Variant '{variant}' is not supported.")
    config = configurations[variant]
    return RepVGG(
        num_blocks=config['num_blocks'],
        num_classes=num_classes,
        width_multiplier=config['width_multiplier']
    )
