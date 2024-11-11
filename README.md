# Deep Learning Models in PyTorch 🚀

**In-depth, annotated implementations of popular deep learning models** – tailored for clear understanding and optimized for performance on the M2 MacBook Air.

---

## What's Inside 

Each model comes with **intensive code annotations** to share insights into the architecture, training tricks, and reasoning behind design choices. Dive in to learn, adapt, and experiment!

### Models Implemented:

- **ResNet** 🖼️  
  - **Dataset**: CIFAR-10  
  - Leveraging ResNet to tackle image classification, with optimizations for running on the M2 MacBook Air.

- **Transformer** 🔄  
  - **Architecture**: Simplified, with reduced parameters  
  - Designed to fit the M2 MacBook Air’s performance, this reduced Transformer model retains the essence of the architecture without overwhelming your device.

- **Deep Q-Network (DQN)** 🎢  
  - **Environment**: CartPole-v1  
  - Implements DQN for CartPole, avoiding convolutional layers (ideal for CartPole’s simple state space) to keep things lean and efficient.

- **AutoEncoder** 🧩  
  - **Dataset**: CIFAR-10  
  - Trains an AutoEncoder for image reconstruction, showcasing a robust dimensionality reduction model on CIFAR-10.

- **Variational AutoEncoder (VAE)** 🔍  
  - **Dataset**: MNIST  
  - A compact VAE designed for the MNIST dataset, making it accessible for experimenting with generative modeling techniques.

- **RepVGG** 🔥  
  - **Dataset**: CIFAR-100  
  - A customized RepVGG model adapted specifically for CIFAR-100. We made several architectural adjustments and applied additional training techniques to optimize the model for CIFAR-100:
    - **Stage Depths**: Reduced the number of blocks per stage to `[1, 2, 4, 6, 1]`, compared to the deeper structure required for larger datasets like ImageNet. This change lowers the model’s parameters and training time, making it more efficient for CIFAR-100.
    - **Width Multipliers**: Scaled down the width multipliers to `[0.75, 0.75, 0.75, 1]`, which controls the number of channels per stage and reduces the model's size.
    - **Data Augmentation (AutoAugment)**: Implemented AutoAugment, which applies advanced data augmentation policies during training. This boosts model generalization and accuracy by providing diverse image transformations, which is particularly beneficial for smaller datasets like CIFAR-100.
    - **Regularization Techniques**: Added label smoothing and mixup, further enhancing generalization and preventing overfitting on CIFAR-100.
    - **Global Average Pooling and Final Classifier**: Added a global average pooling layer before the final fully connected layer to aggregate features for CIFAR-100’s 100 classes.
  - **Performance**: This modified RepVGG model achieves approximately **74% accuracy on the CIFAR-100 test set**. The combination of architectural adjustments and enhanced training techniques makes it well-suited for the M2 MacBook Air.

---
