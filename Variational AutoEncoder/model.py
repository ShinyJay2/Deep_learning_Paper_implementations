# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE is Encoder - Latent Space - Decoder structure.
# In this implementation, we enhance the VAE with:
# - Convolutional layers to capture spatial hierarchies in MNIST images.
# - Deeper networks for increased capacity.
# - Separate networks for mu and logvar for specialized learning.
# - Batch Normalization and Dropout for regularization.
# - KL Annealing during training to balance reconstruction and regularization.
# - Variational Mean Field and Variational Flow for more expressive approximate posteriors.

# The components are:
# - Encoder: Encodes the input images into latent representations (mu and logvar).
# - Reparameterization: Samples latent variable z from the latent distribution.
# - Variational Flow: Transforms z to a more expressive distribution.
# - Decoder: Reconstructs the input images from the latent variable z.

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        # What parameters do we need for the Encoder?
        # Since we're processing images, we define convolutional layers to extract features,
        # and finally map to the latent dimension (latent_dim).
        # We don't need an output dimension because we're outputting mu and logvar (log variance),
        # which parameterize q_phi(z|x), the approximate posterior of p(z|x).

        # For MNIST, input images are 1x28x28.
        # Define convolutional layers to capture spatial features.
        # We progressively increase the number of channels while reducing spatial dimensions.

        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # Output: 32 x 14 x 14
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm helps with training stability.

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # Output: 64 x 7 x 7
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: 128 x 7 x 7
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.4)  # Dropout to prevent overfitting.

        # Flatten the output from conv layers before feeding to fully connected layers.
        # The output of conv3 is (batch_size, 128, 7, 7), so we flatten to (batch_size, 128 * 7 * 7).
        self.flatten = nn.Flatten()

        # Utilize separate networks for mu and logvar to allow specialized learning.
        # Each has its own fully connected layers.

        # What is the meaning of outputting mean and variance of each feature (latent variable)?
        # Imagine that the latent space has variables capturing different characteristics of the digits
        # (like thickness, tilt, style, etc.).
        # While thickness itself doesn't "behave like a Gaussian," the latent space embedding of thickness variations
        # can approximate a Gaussian distribution.
        # This means: Most of the digit images cluster around certain values for thickness and tilt
        # (e.g., centered around "average thickness" or "straight alignment").
        # Extreme variations are less common, resulting in a distribution that is "bell-shaped" around these typical values.

        # Network for mu (mean of the latent Gaussian distribution).
        self.fc_mu = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, latent_dim)  # Output dimension is the latent dimension.
        )

        # Network for logvar (log variance of the latent Gaussian distribution).
        self.fc_logvar = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        # x is of shape (batch_size, 1, 28, 28)
        # Pass through convolutional layers with ReLU activation and BatchNorm.
        x = F.relu(self.bn1(self.conv1(x)))  # Output: (batch_size, 32, 14, 14)
        x = F.relu(self.bn2(self.conv2(x)))  # Output: (batch_size, 64, 7, 7)
        x = F.relu(self.bn3(self.conv3(x)))  # Output: (batch_size, 128, 7, 7)
        x = self.dropout(x)  # Apply dropout.

        x = self.flatten(x)  # Flatten to (batch_size, 128 * 7 * 7).

        # Obtain mu and logvar for the latent Gaussian distribution.
        mu = self.fc_mu(x)       # Output: (batch_size, latent_dim)
        logvar = self.fc_logvar(x)  # Output: (batch_size, latent_dim)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        # Our VAE aim is to reconstruct data (in this case MNIST) from our latent variable z.
        # How is our latent variable z constructed?
        # Encoder outputs two vectors (mu and logvar), each of size latent_dim.
        # How do we sample z?
        # We sample z by taking the mean μ, adding the standard deviation σ (derived from logvar),
        # and multiplying by a random number ε drawn from a standard Gaussian (i.e., N(0,1) for each dimension).

        # Upsample from latent_dim to 128 * 7 * 7 to match the shape before deconvolutions.
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 128 * 7 * 7),
            nn.ReLU(),
        )

        # Define transpose convolutional layers (deconvolutions) to reconstruct the image.
        # We reverse the operations of the encoder.

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)  # Output: 64 x 7 x 7
        self.bn1 = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Output: 32 x 14 x 14
        self.bn2 = nn.BatchNorm2d(32)

        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # Output: 1 x 28 x 28

        self.dropout = nn.Dropout(0.4)  # Dropout for regularization.

    def forward(self, z):
        # Decode from latent space to reconstructed image.
        x = self.fc(z)  # Map latent vector to the shape suitable for deconvolution.
        x = x.view(-1, 128, 7, 7)  # Reshape to (batch_size, 128, 7, 7).

        # Pass through deconvolutional layers with ReLU activation and BatchNorm.
        x = F.relu(self.bn1(self.deconv1(x)))  # Output: (batch_size, 64, 7, 7)
        x = F.relu(self.bn2(self.deconv2(x)))  # Output: (batch_size, 32, 14, 14)
        x = self.dropout(x)  # Apply dropout.

        x = torch.sigmoid(self.deconv3(x))  # Output: (batch_size, 1, 28, 28)
        # Sigmoid activation to get pixel values between 0 and 1.
        return x

# Implementing Variational Mean Field
class VariationalMeanField(nn.Module):
    def __init__(self):
        super().__init__()

        # In basic VAE, the approximated p(z|x), which refers to q_phi(z|x), is assumed to be a simple Gaussian.
        # This is called the Mean-field Gaussian approximation.
        # q_phi(z|x) ~ N ( mu(x), diag(sigma(x)^2) )

    def sample_z(self, mu, logvar):
        # Reparameterization trick:
        # Instead of sampling z ~ N(mu, sigma^2), we sample epsilon ~ N(0, I)
        # and compute z = mu + epsilon * sigma.

        # what is torch.randn_like?
        # torch.randn_like is a PyTorch function that generates a tensor of random numbers
        # sampled from a standard normal distribution (mean 0, variance 1)
        # with the same shape as the input tensor provided.

        std = torch.exp(0.5 * logvar)  # Compute standard deviation from log variance.
        epsilon = torch.randn_like(std)  # Sample epsilon from standard normal distribution.
        z = mu + epsilon * std  # Return reparameterized z.
        return z

# Implementing Variational Flow using Planar Flows
class PlanarFlow(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Planar flow parameters: u, w, and b.
        # These are trainable parameters that define the flow transformation.
        # They shape the approximate posterior to be more flexible than a simple Gaussian.

        self.u = nn.Parameter(torch.randn(1, latent_dim))  # Shape: (1, latent_dim)
        self.w = nn.Parameter(torch.randn(1, latent_dim))  # Shape: (1, latent_dim)
        self.b = nn.Parameter(torch.zeros(1))              # Scalar bias term

    def forward(self, z):
        # z: Input latent variable, shape (batch_size, latent_dim)

        # Planar flow transformation:
        # f(z) = z + u * h(w^T z + b)
        # where h is an element-wise nonlinearity, typically tanh.

        # Compute the linear term w^T z + b
        linear = torch.matmul(z, self.w.t()) + self.b  # Shape: (batch_size, 1)

        # Apply the nonlinearity h (tanh in this case)
        h = torch.tanh(linear)  # Shape: (batch_size, 1)

        # Compute the flow transformation f(z)
        f_z = z + self.u * h  # Shape: (batch_size, latent_dim)

        # Compute the derivative h' for the Jacobian determinant
        h_prime = 1 - h ** 2  # Derivative of tanh is (1 - tanh^2), shape: (batch_size, 1)

        # Compute the term psi = h' * w
        psi = h_prime * self.w  # Element-wise multiplication, shape: (batch_size, latent_dim)

        # Compute u^T psi^T
        u_psi = torch.matmul(psi, self.u.t())  # Shape: (batch_size, 1)

        # Compute determinant of the Jacobian
        det_jacobian = 1 + u_psi  # Shape: (batch_size, 1)

        # Ensure the determinant is positive
        det_jacobian = torch.abs(det_jacobian)  # Take absolute value to avoid negative determinants

        # Compute the log determinant of the Jacobian
        log_det_jacobian = torch.log(det_jacobian + 1e-10)  # Adding epsilon for numerical stability

        # Return the transformed z and the log determinant
        return f_z, log_det_jacobian.squeeze(-1)  # Squeeze only the last dimension

class VariationalFlow(nn.Module):
    def __init__(self, latent_dim, flow_length=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.flow_length = flow_length

        # Initialize a list of flow transformations (Planar Flows)
        self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(flow_length)])

        # Initialize the mean-field approximation to sample z0
        self.mean_field = VariationalMeanField()

    def forward(self, mu, logvar):
        # Sample z0 from the mean-field Gaussian q0(z|x) ~ N(mu, sigma^2)
        z = self.mean_field.sample_z(mu, logvar)  # Shape: (batch_size, latent_dim)

        sum_log_det_jacobians = torch.zeros(z.size(0)).to(z.device)  # Shape: (batch_size,)

        # Apply each flow transformation sequentially
        for flow in self.flows:
            # Each flow transforms z and returns the log determinant of its Jacobian
            z, log_det_jacobian = flow(z)  # z is updated, log_det_jacobian is (batch_size,)
            # Accumulate the log determinant
            sum_log_det_jacobians += log_det_jacobian  # Element-wise addition

        # After applying all flows, z is the transformed latent variable
        # sum_log_det_jacobians is the total adjustment to the log probability due to the flows

        return z, sum_log_det_jacobians  # Return transformed z and total log determinant

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim, use_flow=False, flow_length=5):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.use_flow = use_flow

        if self.use_flow:
            # Initialize Variational Flow for more expressive posterior
            self.variational_flow = VariationalFlow(latent_dim, flow_length)
        else:
            # Use Mean Field approximation
            self.mean_field = VariationalMeanField()

    def forward(self, x):
        # Encode input to obtain mu and logvar
        mu, logvar = self.encoder(x)

        # Why is p(z|x) called a "Posterior"?
        # Because it represents our "after-observation" knowledge about z given the observed x.
        # The components are:
        # - Prior p(z): This is our initial belief or assumption about the distribution of the latent variable z,
        #   independent of any observed data x.
        # - Likelihood p(x|z): Probability of observing x given latent z.
        # - Posterior p(z|x): After observing data x, updated belief of our latent variable distribution.

        if self.use_flow:
            # Use Variational Flow for more expressive posterior
            z, sum_log_det_jacobians = self.variational_flow(mu, logvar)
            # Decode z to reconstruct x
            x_recon = self.decoder(z)
            return x_recon, mu, logvar, sum_log_det_jacobians
        else:
            # Use Mean Field approximation
            z = self.mean_field.sample_z(mu, logvar)
            # Decode z to reconstruct x
            x_recon = self.decoder(z)
            return x_recon, mu, logvar
