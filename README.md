# Deep Learning Models in PyTorch 🚀

**In-depth, annotated implementations of popular deep learning models** – tailored for clear understanding and optimized for performance on the M2 MacBook Air and high-performance desktop setups.

---

## What's Inside 

Each model comes with **intensive code annotations** to share insights into the architecture, training tricks, and reasoning behind design choices. Dive in to learn, adapt, and experiment!

### Models Implemented:

#### **Supervised Learning**

- **ResNet** 🖼️  
  - **Dataset**: CIFAR-10  
  - Leverages ResNet to tackle image classification, with optimizations for running on the M2 MacBook Air.

- **Transformer** 🔄  
  - **Architecture**: Simplified, with reduced parameters  
  - Designed to fit the M2 MacBook Air’s performance, this reduced Transformer model retains the essence of the architecture without overwhelming your device.

- **RepVGG** 🔥  
  - **Dataset**: CIFAR-100  
  - Adapted for CIFAR-100 with reduced depth, lower width multipliers, and enhanced training techniques like AutoAugment, label smoothing, and mixup, achieving around **74% accuracy**.

#### **Reinforcement Learning**

- **Deep Q-Network (DQN)** 🎢  
  - **Environment**: CartPole-v1  
  - Implements DQN for CartPole, avoiding convolutional layers (ideal for CartPole’s simple state space) to keep things lean and efficient.

- **Proximal Policy Optimization (PPO)** 🎯  
  - **Environment**: HalfCheetah-v4  
  - **Description**: Implements PPO for continuous control in the HalfCheetah-v4 environment, leveraging efficient policy optimization techniques to achieve robust performance. Optimized for both the M2 MacBook Air and high-performance desktop setups, ensuring smooth training and evaluation processes.

#### **Unsupervised Learning**

- **AutoEncoder** 🧩  
  - **Dataset**: CIFAR-10  
  - Trains an AutoEncoder for image reconstruction, showcasing a robust dimensionality reduction model on CIFAR-10.

- **Variational AutoEncoder (VAE)** 🔍  
  - **Dataset**: MNIST  
  - A compact VAE designed for the MNIST dataset, making it accessible for experimenting with generative modeling techniques.

#### **Self-Supervised Learning**

- **SimCLR** 🔍  
  - **Dataset**: Tiny ImageNet  
  - Implements SimCLR’s contrastive learning framework on Tiny ImageNet. This model uses a ResNet backbone with a projection head to learn representations in an unsupervised manner, optimized for a high-performance desktop setup with an RTX 4060 GPU and 64GB of RAM.

---
