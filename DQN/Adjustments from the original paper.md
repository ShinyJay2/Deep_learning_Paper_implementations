# Reduced Deep Q-Network (DQN) for CartPole 🎢

This implementation of Deep Q-Network (DQN) is optimized specifically for the **CartPole** environment on limited-resource devices, like the M2 MacBook Air with 8GB memory. Given the computational constraints, we’ve simplified the original DQN architecture to better suit the lower-dimensional CartPole environment.

## Why CartPole instead of Atari Pong?

The original DQN architecture was designed to handle high-dimensional environments like Atari Pong, which processes raw pixel images (84x84 grayscale) and requires a Convolutional Neural Network (CNN) to extract spatial features. However, Atari Pong is computationally heavy, especially for devices with limited memory and processing power.

For CartPole:
- **Input Type**: CartPole’s state is represented by a low-dimensional vector (pole angles, cart location, velocity), so it doesn’t require complex spatial feature extraction.
- **Simplified Architecture**: Using a fully connected layer (Multi-Layer Perceptron or MLP) is sufficient to map the state variables directly to Q-values for actions, reducing computational requirements.

## Architectural Adjustments for CartPole

- **Network Type**: Instead of the convolutional layers in the original DQN, this reduced version uses a fully connected architecture.
- **Layer Composition**: Our model consists of 3 fully connected (FC) layers, enough to handle CartPole’s structured state variables.
- **Parameter Count**: The absence of convolutional layers significantly reduces the parameter count, making the model lightweight and suitable for CartPole’s simple environment.

## Visual Summary of Differences

The table below summarizes the key distinctions between the Original DQN (Mnih et al., 2015) and this Reduced DQN for CartPole:

| Aspect                | Original DQN (Mnih et al., 2015)         | Reduced DQN (CartPole)           |
|-----------------------|------------------------------------------|----------------------------------|
| **Input Type**        | Raw pixel images (84x84x4 grayscale)     | Structured state variables (4D)  |
| **Network Type**      | Convolutional Neural Network (CNN)       | Multi-Layer Perceptron (MLP)     |
| **Layer Composition** | 3 Conv layers + 2 FC layers             | 3 FC layers                      |
| **Number of Layers**  | 5 total layers                           | 3 total layers                   |
| **Activation Functions** | ReLU                                | ReLU                             |
| **Output Layer Size** | High (e.g., 18 actions for Atari)       | Low (e.g., 2 actions for CartPole) |
| **Purpose of Architecture** | Extract spatial features from images | Map state variables to Q-values |
| **Parameter Count**   | High due to multiple Conv layers         | Lower, suitable for simple tasks |
| **Suitability**       | Complex, high-dimensional tasks          | Simple, low-dimensional tasks    |

### Additional Notes

- **Activation Functions**: Both architectures use ReLU activation for non-linearity, which is effective for DQN.
- **Output Layer**: The output layer size depends on the action space. For Atari environments, it can be large (e.g., 18 actions), but for CartPole, only 2 actions (left or right) are needed.
- **Reduced Parameter Complexity**: By using an MLP, this DQN version has significantly fewer parameters, making it more efficient and faster to train on devices with limited computational power, while remaining effective for simple, structured environments like CartPole.

This architecture modification makes the Reduced DQN model ideal for running on an M2 MacBook Air or similar resource-constrained devices without compromising performance for the CartPole task.
