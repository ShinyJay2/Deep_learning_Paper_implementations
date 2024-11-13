# SimCLR Implementation on Tiny ImageNet

This repository provides an implementation of SimCLR, a contrastive learning framework, using Tiny ImageNet as the dataset. SimCLR learns visual representations in an unsupervised manner by maximizing agreement between different augmented views of the same image. This repository is optimized for a desktop with an RTX 4060 GPU and 64GB of RAM, making it suitable for efficient training on Tiny ImageNet.

## Repository Structure

- `train.py` — Main training script that controls data loading, model training, and loss computation. 
- `model.py` — Defines the SimCLR model architecture, including the ResNet backbone and the projection head.
- `augmentations.py` — Contains data augmentations specific to SimCLR, such as random cropping, color jitter, and Gaussian blur. 
- `nt_xent_loss.py` — Implements the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.

## How the Model Works

### High-Level Overview

SimCLR leverages contrastive learning to train the model to distinguish between different augmented views of images. The main idea is to pull together representations of augmented views from the same image and push apart representations from different images. The framework operates as follows:

1. **Data Augmentation**: Each image in a batch is augmented twice, resulting in two different “views” of the same image. These pairs of views are considered "positive pairs," while the rest of the images in the batch are treated as "negative examples."
  
2. **ResNet Backbone**: Each augmented view is passed through a ResNet model (e.g., ResNet-50) to obtain a high-dimensional feature representation, `h`. The ResNet model used here is a backbone network without its final classification layer, providing feature vectors instead of class predictions.

3. **Projection Head**: The high-dimensional feature vector, `h`, is further processed by a projection head, a two-layer MLP. The projection head outputs a lower-dimensional vector, `z`, which is optimized for the contrastive loss.

4. **Contrastive Loss (NT-Xent)**: The contrastive loss maximizes the similarity between positive pairs (views from the same image) while minimizing the similarity between negative pairs (views from different images). This is done in the projection space, where `z` vectors are compared.

### Detailed Components

- **ResNet Backbone**: The ResNet model is used as a feature extractor and provides a rich, high-dimensional representation, `h`, of each augmented view. The final classification layer of ResNet is removed, as we are not performing supervised classification in this self-supervised setup.

- **Projection Head**: 
  - The projection head maps the high-dimensional representation `h` from the ResNet backbone into a lower-dimensional space, producing `z`. This transformation involves a two-layer MLP with a hidden ReLU activation and an output layer that matches the contrastive loss dimension (typically 128).
  - **Why the Projection Head?** The projection head improves the effectiveness of the contrastive learning by mapping `h` into a space better suited for the contrastive loss. In SimCLR, applying the contrastive loss directly on `h` (the feature space) was found to be less effective. The projection head outputs `z`, a more compact representation that encourages learning meaningful representations for downstream tasks.

### Summary of `h` and `z`
- **`h`**: The high-dimensional feature vector output from the ResNet backbone, capturing a complex representation of the image.
- **`z`**: The low-dimensional projection from the MLP, optimized for contrastive loss. The contrastive loss is applied to `z`, not `h`, for better representation learning.

## Hardware and Batch Size

Implemented with an RTX 4060 GPU and 64GB of RAM, you can use batch sizes around 512 or higher to accelerate training while ensuring the GPU memory is optimally utilized.

## Notes

- This implementation follows the structure of the original SimCLR paper to maintain model performance on self-supervised tasks.
- Checkpoints and logs can be set up in `train.py` to save model states and track training progress.

## References

SimCLR Paper: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) by Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton.

--- 

This `README.md` should provide a comprehensive guide for setting up and understanding the SimCLR implementation. Let me know if you need further customizations!
