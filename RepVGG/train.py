import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import create_RepVGG  # Ensure model.py is accessible
import matplotlib.pyplot as plt
import numpy as np
import random

# Advanced Data Augmentation Imports
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
# Note: If you wish to use Cutout or other custom augmentations, you'll need to import or define them.

if __name__ == '__main__':
    # Set the device to MPS if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device:", device)
    else:
        device = torch.device("cpu")
        print("MPS device not found. Using CPU.")

    # Hyperparameters
    num_epochs = 100  # Increased number of epochs for better training
    batch_size = 128  # Adjusted batch size for better utilization
    learning_rate = 0.1  # Increased initial learning rate
    momentum = 0.9
    weight_decay = 5e-4

    # Seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Data augmentation and normalization for training
    # Advanced data augmentation techniques included
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Randomly crop and pad images
        transforms.RandomHorizontalFlip(),     # Randomly flip images horizontally
        # Advanced augmentation techniques
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),  # AutoAugment policy for CIFAR-10
        # Cutout or Random Erasing can be added here if desired
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                             std=(0.2675, 0.2565, 0.2761)),  # Normalize using CIFAR-100 mean and std
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                             std=(0.2675, 0.2565, 0.2761)),
    ])

    # Load CIFAR-100 dataset
    # The dataset will be downloaded if not already present in the specified root directory
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # DataLoader wraps an iterable around the Dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # Initialize the model
    model = create_RepVGG(variant='CIFAR', num_classes=100)

    # Move the model to the specified device
    model.to(device)

    # Define loss function and optimizer
    # Using CrossEntropyLoss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing to improve generalization
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)
    # Cosine annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training and validation
    best_acc = 0  # Best test accuracy
    save_path = 'repvgg_cifar_best.pth'

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch [{epoch}/{num_epochs}]')

        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(f'  Train Step: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}% ({correct}/{total})')

        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader, start=1):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        print(f'  Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {acc:.2f}% ({correct}/{total})')

        # Save checkpoint if best accuracy
        if acc > best_acc:
            print('  Saving new best model...')
            best_acc = acc
            torch.save(model.state_dict(), save_path)

        scheduler.step()

    print('Training completed.')

    # Fuse the model before inference
    print('Fusing the model...')
    model.fuse()

    # Save the fused model
    fused_save_path = 'repvgg_cifar_fused.pth'
    torch.save(model.state_dict(), fused_save_path)
    print('Fused model saved.')

    # ================== Added Code for Testing and Visualization ==================

    # Define CIFAR-100 class names
    classes = test_dataset.classes

    # Function to display images with predictions
    def show_predictions(model, device, test_loader, classes, num_images=5):
        model.eval()
        dataiter = iter(test_loader)
        images, labels = next(dataiter)

        images = images[:num_images]
        labels = labels[:num_images]

        # Move to device
        images = images.to(device)
        labels = labels.to(device)

        # Get predictions
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Move images and labels back to CPU for plotting
        images = images.cpu()
        labels = labels.cpu()
        predicted = predicted.cpu()

        # Unnormalize images
        mean = np.array([0.5071, 0.4867, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])
        images = images.permute(0, 2, 3, 1).numpy()
        images = std * images + mean  # Unnormalize
        images = np.clip(images, 0, 1)

        # Plot images with predicted labels
        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
        for idx in range(num_images):
            ax = axes[idx]
            ax.imshow(images[idx])
            ax.set_title(f'Predicted: {classes[predicted[idx]]}\nTrue: {classes[labels[idx]]}')
            ax.axis('off')
        plt.show()

    # Display some test images with predictions
    print('Displaying some test images with predicted labels...')
    show_predictions(model, device, test_loader, classes, num_images=5)
