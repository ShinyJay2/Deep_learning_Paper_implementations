# train.py
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from model import VariationalAutoencoder

# Function to compute loss.
def loss_function(recon_x, x, mu, logvar, kl_weight, sum_log_det_jacobians=None):
    # Reconstruction loss measures how well the output matches the input.
    # For images, we can use Binary Cross-Entropy (BCE) loss.
    # recon_x and x are of shape (batch_size, 1, 28, 28).
    # We need to flatten them to (batch_size, 28*28).

    recon_x = recon_x.view(-1, 28*28)
    x = x.view(-1, 28*28)

    # Compute BCE loss between reconstructed and original images.
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # Compute element-wise KLD (shape: [batch_size])
    KLD_elementwise = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    # Adjust KL Divergence for Variational Flow
    if sum_log_det_jacobians is not None:
        # When using flows, the true posterior is transformed by the flows,
        # so we need to adjust the KL divergence accordingly.
        KLD_elementwise -= sum_log_det_jacobians

    # Sum over the batch to get the total KLD
    KLD = torch.sum(KLD_elementwise)

    # Apply KL Annealing: Multiply KL divergence with kl_weight.
    # This helps the model to focus on reconstruction in the initial epochs.

    return BCE + kl_weight * KLD, BCE, KLD  # Return total loss and individual components.

def train(model, train_loader, optimizer, device, epochs, kl_annealing_steps):
    model.train()  # Set model to training mode.
    train_loss = []  # List to store training loss per epoch.
    for epoch in range(1, epochs + 1):
        total_loss = 0  # Total loss for the epoch.
        total_bce = 0   # Total reconstruction loss for the epoch.
        total_kld = 0   # Total KL divergence loss for the epoch.
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)  # Move data to the device (CPU or GPU).

            optimizer.zero_grad()  # Zero the gradients.

            if model.use_flow:
                recon_batch, mu, logvar, sum_log_det_jacobians = model(data)
                # KL Annealing: Increase kl_weight over epochs.
                kl_weight = min(1.0, epoch / kl_annealing_steps)
                # Compute loss.
                loss, bce, kld = loss_function(recon_batch, data, mu, logvar, kl_weight, sum_log_det_jacobians)
            else:
                recon_batch, mu, logvar = model(data)
                kl_weight = min(1.0, epoch / kl_annealing_steps)
                loss, bce, kld = loss_function(recon_batch, data, mu, logvar, kl_weight)

            loss.backward()  # Backward pass.
            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()
            optimizer.step()  # Update model parameters.

            if batch_idx % 100 == 0:
                # Print training status every 100 batches.
                print(f'Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item()/len(data):.4f} BCE: {bce.item()/len(data):.4f} KLD: {kld.item()/len(data):.4f} kl_weight: {kl_weight:.4f}')

        # Compute average losses for the epoch.
        avg_loss = total_loss / len(train_loader.dataset)
        avg_bce = total_bce / len(train_loader.dataset)
        avg_kld = total_kld / len(train_loader.dataset)
        train_loss.append(avg_loss)  # Append average loss to the list.
        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} BCE: {avg_bce:.4f} KLD: {avg_kld:.4f}')
    return model, train_loss  # Return the trained model and training losses.

def test(model, test_loader, device):
    # Evaluate the model on the test set.
    model.eval()  # Set model to evaluation mode.
    test_loss = 0
    total_bce = 0
    total_kld = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            if model.use_flow:
                recon_batch, mu, logvar, sum_log_det_jacobians = model(data)
                loss, bce, kld = loss_function(recon_batch, data, mu, logvar, kl_weight=1.0, sum_log_det_jacobians=sum_log_det_jacobians)
            else:
                recon_batch, mu, logvar = model(data)
                loss, bce, kld = loss_function(recon_batch, data, mu, logvar, kl_weight=1.0)
            test_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

    # Compute average losses for the test set.
    avg_loss = test_loss / len(test_loader.dataset)
    avg_bce = total_bce / len(test_loader.dataset)
    avg_kld = total_kld / len(test_loader.dataset)
    print(f'====> Test set loss: {avg_loss:.4f} BCE: {avg_bce:.4f} KLD: {avg_kld:.4f}')
    return avg_loss

def evaluate_performance(model, test_loader, device):
    # Evaluate the model's reconstruction performance on the test set.
    model.eval()
    total_bce = 0
    total_kld = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            if model.use_flow:
                recon_batch, mu, logvar, sum_log_det_jacobians = model(data)
                _, bce, kld = loss_function(recon_batch, data, mu, logvar, kl_weight=1.0, sum_log_det_jacobians=sum_log_det_jacobians)
            else:
                recon_batch, mu, logvar = model(data)
                _, bce, kld = loss_function(recon_batch, data, mu, logvar, kl_weight=1.0)
            total_bce += bce.item()
            total_kld += kld.item()

    avg_bce = total_bce / len(test_loader.dataset)
    avg_kld = total_kld / len(test_loader.dataset)
    print(f'====> Performance Evaluation on Test Set: BCE: {avg_bce:.4f} KLD: {avg_kld:.4f}')

def visualize_reconstruction(model, test_loader, device):
    # Visualize reconstructed images.
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))  # Get a batch of test images.
        data = data.to(device)
        if model.use_flow:
            recon_batch, mu, logvar, _ = model(data)
        else:
            recon_batch, mu, logvar = model(data)
        n = min(data.size(0), 8)  # Number of images to display.

        # Concatenate original and reconstructed images for comparison.
        comparison = torch.cat([data[:n],
                                recon_batch[:n]])

        # Save the images to a file.
        save_image(comparison.cpu(), 'reconstruction.png', nrow=n)
        print('Reconstructed images saved to reconstruction.png')

def visualize_generation(model, device, latent_dim):
    # Generate new samples from the latent space.
    model.eval()
    with torch.no_grad():
        # Sample latent vectors from the standard normal distribution N(0, I).
        sample = torch.randn(64, latent_dim).to(device)
        sample = model.decoder(sample)  # Decode the latent vectors to images.

        # Save the generated images to a file.
        save_image(sample.cpu(), 'sample.png')
        print('Generated images saved to sample.png')

def interpolate_latent_space(model, device, latent_dim):
    # Interpolate between two random points in the latent space.
    model.eval()
    with torch.no_grad():
        # Select two random points from the latent space.
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)

        # Number of interpolation steps.
        num_steps = 10

        # Generate interpolation coefficients between 0 and 1.
        alpha = torch.linspace(0, 1, steps=num_steps).unsqueeze(1).to(device)

        # Interpolate between z1 and z2.
        z_interp = (1 - alpha) * z1 + alpha * z2  # Shape: (num_steps, latent_dim)

        # Decode the interpolated latent vectors.
        interp_images = model.decoder(z_interp)

        # Save the interpolated images to a file.
        save_image(interp_images.cpu(), 'interpolation.png', nrow=num_steps)
        print('Interpolated images saved to interpolation.png')

def visualize_latent_space(model, test_loader, device):
    # Visualize the 2D latent space (only if latent_dim == 2).
    model.eval()
    latent_space = []
    labels_list = []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            mu, logvar = model.encoder(data)
            if model.use_flow:
                z, _ = model.variational_flow(mu, logvar)
            else:
                z = model.mean_field.sample_z(mu, logvar)
            latent_space.append(z.cpu().numpy())
            labels_list.append(labels.numpy())
    latent_space = np.concatenate(latent_space, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    # Plot the latent space with points colored by their labels.
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(latent_space[:, 0], latent_space[:, 1], c=labels_list, cmap='tab10', s=2)
    plt.colorbar(scatter)
    plt.xlabel('Latent dimension 1')
    plt.ylabel('Latent dimension 2')
    plt.title('Latent space visualization')
    plt.savefig('latent_space.png')
    plt.show()
    print('Latent space visualization saved to latent_space.png')

def save_image(tensor, filename, nrow=8):
    from torchvision.utils import save_image
    save_image(tensor, filename, nrow=nrow)

def main():
    # Set hyperparameters.
    batch_size = 128
    epochs = 10
    learning_rate = 1e-3
    latent_dim = 10  # Set to 2 if you want to visualize latent space.
    kl_annealing_steps = 5  # Number of epochs over which to anneal KL weight.
    use_flow = True  # Set to True to use Variational Flow
    flow_length = 5  # Number of flow steps

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available.

    # Load MNIST dataset.
    # Apply necessary transformations to the data.
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize the images to [-1, 1] if using tanh activation in the decoder.
        # transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                                  transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the VAE model.
    # MNIST images are 1x28x28.
    model = VariationalAutoencoder(latent_dim=latent_dim, use_flow=use_flow, flow_length=flow_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model.
    model, train_loss = train(model, train_loader, optimizer, device, epochs, kl_annealing_steps)

    # Test the model.
    test_loss = test(model, test_loader, device)

    # Evaluate performance on the test set.
    evaluate_performance(model, test_loader, device)

    # Save the trained model parameters.
    torch.save(model.state_dict(), 'vae.pth')
    print('Model saved to vae.pth')

    # Visualize reconstructed images.
    visualize_reconstruction(model, test_loader, device)

    # Visualize generated samples from the latent space.
    visualize_generation(model, device, latent_dim)

    # Interpolate between two random latent vectors and visualize.
    interpolate_latent_space(model, device, latent_dim)

    # Visualize latent space if latent_dim == 2.
    if latent_dim == 2:
        visualize_latent_space(model, test_loader, device)

    # Plot training loss over epochs.
    plt.figure()
    plt.plot(range(1, epochs + 1), train_loss, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()
    print('Training loss plot saved to training_loss.png')

if __name__ == '__main__':
    main()
