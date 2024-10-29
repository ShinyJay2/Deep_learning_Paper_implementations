import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import AutoEncoder
import matplotlib.pyplot as plt
import numpy as np

# Transformations for the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='~/Desktop/Model_implementation/data', train=True,
                                        download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoEncoder().to(device)
# Using MSE Loss for continuous-valued data like RGB images
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Using a learning rate scheduler to reduce the learning rate over time for better convergence
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Training loop
# Increasing the number of epochs to allow the model to converge better
epochs = 50
for epoch in range(epochs):
    running_loss = 0.0
    for data in trainloader:
        inputs, _ = data
        inputs = inputs.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        latent, outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    # Step the learning rate scheduler
    scheduler.step()

        # Print statistics
    running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}')

print('Finished Training')

# Save the model
torch.save(model.state_dict(), 'autoencoder_cifar10.pth')

# Testing the model
model.load_state_dict(torch.load('autoencoder_cifar10.pth', map_location=device, weights_only=True))
model.eval()

testset = torchvision.datasets.CIFAR10(root='~/Desktop/Model_implementation/data', train=False,
                                       download=False, transform=transform)
testloader = DataLoader(testset, batch_size=10, shuffle=True)

dataiter = iter(testloader)
images, labels = next(dataiter)

# Move images to device
images = images.to(device)

# Get encoded and reconstructed images
with torch.no_grad():
    latent, reconstructed = model(images)

# Move images and encoded data back to CPU for visualization
images = images.cpu()
reconstructed = reconstructed.cpu()
latent = latent.cpu()

# Plot original and reconstructed images
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    # Original images
    axes[0, i].imshow(images[i].permute(1, 2, 0))
    axes[0, i].axis('off')
    # Reconstructed images
    axes[1, i].imshow(reconstructed[i].permute(1, 2, 0))
    axes[1, i].axis('off')

plt.show()

# Plot the 2D latent space
latent = latent.view(latent.size(0), -1)  # Flatten the latent representation
latent = latent.numpy()
labels = labels.numpy()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(latent[:, 0], latent[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, ticks=range(10))
plt.title('2D Latent Space Representation of CIFAR-10')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.show()

# Visualize learned filters from the first convolutional layer of the encoder
filters = model.encoder[0].weight.data.cpu()  # Get the filters from the first Conv2d layer
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    if i < filters.shape[0]:
        filter_img = filters[i].permute(1, 2, 0).numpy()
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())  # Normalize to [0, 1]
        ax.imshow(filter_img)
    ax.axis('off')
plt.suptitle('Learned Filters from the First Convolutional Layer')
plt.show()
