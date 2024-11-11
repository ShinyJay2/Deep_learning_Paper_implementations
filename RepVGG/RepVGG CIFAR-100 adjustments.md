# RepVGG - CIFAR-100 Adjustments

This modified RepVGG model is specifically tailored for the CIFAR-100 dataset, with several adjustments to both the architecture and training techniques for improved performance on limited-resource devices like the M2 MacBook Air.

## Architectural Adjustments

1. **Stage Depths**: Reduced the number of blocks per stage to `[1, 2, 4, 6, 1]` (from the original RepVGG configuration) to reduce computational complexity and training time on CIFAR-100.
2. **Width Multipliers**: Scaled down the width multipliers to `[0.75, 0.75, 0.75, 1]` to control the number of channels and lower the parameter count while maintaining adequate model capacity.

## Training Adjustments

1. **AutoAugment**: Applied advanced data augmentation policies through AutoAugment, which helps to diversify the training data and improve model generalization.
2. **Label Smoothing**: Added label smoothing to soften class labels, which helps in reducing overconfidence and improves model robustness.
3. **Mixup**: Incorporated mixup as a regularization technique to blend images and labels, further enhancing model generalization.

## Performance

With these adjustments, the modified RepVGG model achieves approximately **74% accuracy** on the CIFAR-100 test set, striking a balance between performance and efficiency.
