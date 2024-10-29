import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder - conv layers that suppress image into latent vectors
        # Compressing Image into low-dimensional representations (latent space, bottleneck)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=2) # Reducing to 1x1 with latent dimension 32.
            # here, image is reduced into 2x2, so we have 2x2 kernel for 1x1 representation
        )

        # Decoder - upscaling. Exactly same structure but in oppsite direction.

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # output_padding is used to adjust the dimensions of the transposed convolution's output when upsampling, 
            # ensuring that the final feature maps align perfectly with the desired spatial dimensions.
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
            # Sigmoid forces the output to be interpreted as pixel intensity of a grey scale image. 
            # If you remove the sigmoid, the NN will have to learn that all the outputs should be in the range [0, 1]. 
            # The sigmoid might help making the learning process more stable.
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return latent, decoded
    
    
