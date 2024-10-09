import torch
import torch.nn as nn
import torch.nn.functional as F

# We first build a Residual block.
# This Residual block class means just going through 1 block. (e.g. 3x3 conv, 64 -> 3x3 conv, 64 with skip connections)
# Several of these consist the whole ResNet architecture.

class ResidualBlock(nn.Module):
    # We need to __init__ our ResidualBlock with in_channels, out_channels, and stride = 1
    # In the paper, stride = 2 "when the shortcuts go across feature maps of two sizes" this means (e.g. 64 -> 128) 
    # (3.3 Network Architectures)
    def __init__(self, in_channels, out_channels, stride=1): 
        super().__init__()

        # Fundamental is to use 3x3 conv. Keeps going like 3x3 conv, 64 -> 3x3 conv, 128 and so on.
        # And then BatchNorm the conv.
        # Important! nn.Conv2d ouputs feature map(convolutioned image) and stores filter(kernel) params inside
        # Important! nn.BatchNorm2d outputs BatchNormalized feature maps.

        # Further understanding of CNN:
        # If we input image, pass through conv and batchnorm, it will look like:
        # Suppose the image is 5x5

        # [ 0.2, -1.3,  0.1,  0.5, -0.8 ]
        # [ -0.5, 0.9,  -1.1, 1.3, -0.2 ]
        # [ 0.4,  1.0,  -0.7, -0.3, 0.8 ]
        # [ -1.2, 0.3,  1.2,  -0.4, 0.6 ]
        # [ 0.0, -0.1,  0.7,  1.5, -0.6 ]

        # Since we have 64 channels when starting, we stack 64 of these different feature maps as a "Block", and output it.
        # So we are creating a "block" of feature representations.
        # Now I get it when talking about CNN as some kind of cube shape

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Why padding=1 and bias=False ? This is because we want to keep the size of the height & width of feature map. (Padding 1)
        # * Note: padding is filling the empthy, reduced spaces after convolution.
        # * Note : filter is window, feature map is the output of an image when a filter is applied.
        # Because we BatchNorm, "no need for bias terms"

        # Second conv layer (The main idea of ResNet is to shortcut the x -> weight_layer -> relu -> weight_layer -> F(x) + x)
        # x jumps over 2 weight_layers (which conv1 & conv2 in our case)
        # stride should be 1, and since we increased the channels and outputted it, we need to both apply out_channels?
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Important! conv1 has stride=stride while conv2 has stride=1 !!!
        # This is because we might increase the number of channels in the proecedure. 
        # We increase 64 -> 128 in conv1, and downsample image using stride > 1.

        # Now we need a Shortcut. A bypass.
        # We have 2 cases as mentioned in the paper:
        # 1. Identity shortcut (when dimensions are same)
        # 2. Projection shortcut (when dimensions increase)

        if stride != 1 or in_channels != out_channels: # stride would become 2 when dimension increase, also channels would be different
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()


        # Important! About 1x1 convolution, how it works. basically it's a projection.
        # To match these channels in the shortcut, we apply a 1x1 convolution:
        # Input shape: (Batch_Size, 64, H_in, W_in)
        # 1x1 convolution (W_s) has 128 filters, each of size 1x1:
        # This transforms the 64 channels to 128 channels.
        # The spatial dimensions (H_in, W_in) may also be adjusted by using a stride of 2 if downsampling is needed.
        # The output of the 1x1 convolution is now (Batch_Size, 128, H_out, W_out), which matches the shape of F(x).

        
        # We need a forward pass
        # Remember! The process was: x -> weight_layer -> relu -> weight_layer -> F(x) + x

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        # Add residual
        output = output + self.shortcut(x)
        output = F.relu(output)

        return output

# Now we need our ResNet architecture!
    # 1. We want our ResidualBlock to be used in our architecture.
    # 2. We need to specify our architectue (How many ResidualBlocks per layer? How many layers?)
    # 3. What is the number of output class? (In CIFAR10 we have 10 classes to classify)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_class=10):
        super().__init__()

        self.in_channels = 64  # Why do we assign this with self? Because we want to track channel changes dynamically throughout layers.

        # We need an initial convolution layer as described in the paper (3x3 conv, 64 channels, stride=2)
        # However, authors used ImageNet, which is (224 x 224) while I'm going to use CIFAR10 (32x32)
        # In this case, stride=2 would drastically reduce information

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False) # Same padding (=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Building ResNet with ResidualBlocks

        # In the paper, we go from 64 -> 18 -> 256 -> 512, so we need 4 kinds of layers
        # Now we created our stack_layers function, use this to build the 4 layers

        # Important! Understanding OOP.
        # Why do we write the code like self.layer1 = self.stack_layers() ?
        # The self keyword allows stack_layer to access and modify the instance variable self.in_channels
        # It allows each instance of the ResNet class to maintain its own version of variables,
        # such as in_channels and ensures that methods like stack_layer can operate correctly on those instance-specific variables.
        # * Note: using self.sth allows as to access the self variable (In this case, self.in_channels = 64) in the class ResNet

        # We initialize the layers with stride=2 in layers2~4 because we need stride=2 in the first block. 
        # stride changes to stride=1 in stack_layers().

        self.layer1 = self.stack_layers(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.stack_layers(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.stack_layers(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.stack_layers(block, 512, num_blocks[3], stride=2)

        # We need a Fully-connected layer for classification (the final classification)
        # In the paper, after the 3x3 conv, 512 there comes an avg.pool -> fc 1000.
        # 1. Why use pooling? : Reduce dimension of the feature map before going into fc layer.
        # 2. What is Global Average Pooling(GAP)? : Pooling with window size == featue_map size.
        # 3. Advantages of GAP: No additional weight / captures more info than max pool.

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512, num_class)

        # Important!!!!!
        # Breaking down the process
            # 1. The output from self.layer4, we have 512 feature maps, sized 4x4 
                # (32x32 went into 1 layer stride=1, 3 layers of stride=2, so 2^2 is left)
            # 2. After the avgpool, we have 512 scalars
                # AdaptiveAvgPool2d((1,1)) makes a feature_map into a scalar. We have 512 featue_map, so we have 512 scalar
            # 3. Since we have 512 scalar, Our input neuron of fc layer would be 512, output neuron would be the number of classes (10)
                # 512 scalars that you get after the global average pooling operation are derived from a single image.


        # We need to define how will the layers be created.
        # Important! The first block might have stride=2! 
        # Also, the block() means ResidualBlock (we are going to put our class ResidualBlock inside)
        # So stride=1 if stride is not specified. Also pytorch sets stride=1 if not sepecified.

        # What do we need? : out_channel, block, the first block, then append(extend) the other blocks with for loop
        # Also need a num_blocks for the for loop then. 

    def stack_layers(self, block, out_channels, num_blocks, stride):

        # The first layer would start with in_channels=64, output channels with our out_channels param,
        # stride might be different, so we need a stride param to later set this into stride=2.

        # Define a list of layers containing the first block
        
        layers = [block(self.in_channels, out_channels, stride)] # Since ResidualBlock takes in args(in_channels, out_channels, stride)

        # Important! We need to update the self.in_channels as the out_channels in order to keep track of the dimensions inside layers
        self.in_channels = out_channels

        # extend() allows us to add iterables at once to the list.
        # No need to specify stride=1 here.
        # Why for _ in range(1, num_blocks)? Because we already have 1 block at the start, so no need for num_block + 1
        layers.extend(block(out_channels, out_channels) for _ in range(1, num_blocks)) 

        # Now we have our layers stacked in the layers = [], Add them as a whole in nn.Sequential to adress them as a whole.
        # * (asterisk) allows us to unpack the list, adding elements one by one.

        return nn.Sequential(*layers)
        
    def forward(self, x):
    # Forward propagation through the network
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
    # Although `avgpool` reduces the spatial dimensions to `1x1`, the resulting tensor shape is `(Batch_Size, 512, 1, 1)`. 
    # `torch.flatten(out, 1)` removes the `1x1` dimensions, resulting in a shape of `(Batch_Size, 512)`. 
        # Tensor of shape (2, 512, 1, 1):
        # [ [[value_1], [value_2], ..., [value_512]], [[value_1], [value_2], ..., [value_512]] ] # image 1 and image 2
        # Tensor of shape (2, 512):
        # [ [value_1, value_2, ..., value_512], [value_1, value_2, ..., value_512] ] # image 1 and image 2

    # This prepares the tensor to be passed into the fully connected layer.
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out
        

# Function to create ResNet model with different depths

def ResNet18():
    return ResNet(ResidualBlock, [2,2,2,2], num_class=10)  # We set our num_blocks like this, a list of block numbers of each 4 layers

def ResNet34():
    return ResNet(ResidualBlock, [3,4,6,3], num_class=10)





