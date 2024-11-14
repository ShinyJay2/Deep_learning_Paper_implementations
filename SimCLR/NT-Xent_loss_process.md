# NT-Xent Loss (Normalized Temperature-scaled Cross Entropy) - SimCLR

The NT-Xent loss is a contrastive loss function used in SimCLR to learn visual representations by maximizing agreement between positive pairs (two augmented views of the same image) while minimizing similarity with negative pairs (views from different images). This self-supervised learning objective encourages the model to learn meaningful representations without needing labeled data.

## How NT-Xent Loss Works

### 1. Positive and Negative Pairs

- **Positive Pairs**: Each image in a batch is augmented twice to produce two views, which are considered a positive pair. For example, given an original image \( x_1 \), two augmented versions \( x_1 \) and \( x_1' \) form a positive pair.
- **Negative Pairs**: All other samples in the batch that are not the positive pair are treated as negative examples. For instance, for \( x_1 \), the negative pairs are all other augmented views in the batch: \( x_2, x_2', x_3, x_3', \) etc.

The batch structure provides ground-truth knowledge of which pairs are positive, so positive pairs are predefined and don’t rely on similarity scores.

### 2. Similarity Matrix \( S \)

For a batch of \( N \) images, we create two augmented views for each image, resulting in \( 2N \) samples in the batch. We then compute pairwise cosine similarities between each pair of samples, yielding a \( 2N \times 2N \) similarity matrix \( S \), where each entry \( S_{i,j} \) represents the cosine similarity between embeddings \( z_i \) and \( z_j \).

- **Self-similarities** (diagonal entries) are set to zero, as they don’t provide useful information.
- **Positive pairs** are the similarities between each view and its corresponding augmented view.
- **Negative pairs** are the similarities between each view and all other unrelated views.

### Example of the Similarity Matrix

For a batch with 4 images, each with two augmented views, the similarity matrix \( S \) could look like this:

\[
S = \begin{bmatrix}
\textcolor{red}{0} & \text{Neg: } S_{1,2} & \text{Neg: } S_{1,3} & \text{Neg: } S_{1,4} & \textcolor{blue}{\text{Pos: } S_{1,1'}} & \text{Neg: } S_{1,2'} & \text{Neg: } S_{1,3'} & \text{Neg: } S_{1,4'} \\
\text{Neg: } S_{2,1} & \textcolor{red}{0} & \text{Neg: } S_{2,3} & \text{Neg: } S_{2,4} & \text{Neg: } S_{2,1'} & \textcolor{blue}{\text{Pos: } S_{2,2'}} & \text{Neg: } S_{2,3'} & \text{Neg: } S_{2,4'} \\
\text{Neg: } S_{3,1} & \text{Neg: } S_{3,2} & \textcolor{red}{0} & \text{Neg: } S_{3,4} & \text{Neg: } S_{3,1'} & \text{Neg: } S_{3,2'} & \textcolor{blue}{\text{Pos: } S_{3,3'}} & \text{Neg: } S_{3,4'} \\
\text{Neg: } S_{4,1} & \text{Neg: } S_{4,2} & \text{Neg: } S_{4,3} & \textcolor{red}{0} & \text{Neg: } S_{4,1'} & \text{Neg: } S_{4,2'} & \text{Neg: } S_{4,3'} & \textcolor{blue}{\text{Pos: } S_{4,4'}} \\
\textcolor{blue}{\text{Pos: } S_{1',1}} & \text{Neg: } S_{1',2} & \text{Neg: } S_{1',3} & \text{Neg: } S_{1',4} & \textcolor{red}{0} & \text{Neg: } S_{1',2'} & \text{Neg: } S_{1',3'} & \text{Neg: } S_{1',4'} \\
\text{Neg: } S_{2',1} & \textcolor{blue}{\text{Pos: } S_{2',2}} & \text{Neg: } S_{2',3} & \text{Neg: } S_{2',4} & \text{Neg: } S_{2',1'} & \textcolor{red}{0} & \text{Neg: } S_{2',3'} & \text{Neg: } S_{2',4'} \\
\text{Neg: } S_{3',1} & \text{Neg: } S_{3',2} & \textcolor{blue}{\text{Pos: } S_{3',3}} & \text{Neg: } S_{3',4} & \text{Neg: } S_{3',1'} & \text{Neg: } S_{3',2'} & \textcolor{red}{0} & \text{Neg: } S_{3',4'} \\
\text{Neg: } S_{4',1} & \text{Neg: } S_{4',2} & \text{Neg: } S_{4',3} & \textcolor{blue}{\text{Pos: } S_{4',4}} & \text{Neg: } S_{4',1'} & \text{Neg: } S_{4',2'} & \text{Neg: } S_{4',3'} & \textcolor{red}{0} \\
\end{bmatrix}
\]

- **Red (0)**: Self-similarities (e.g., \( S_{1,1}, S_{2,2} \)) — Set to zero as they are not useful for contrastive learning.
- **Blue (Pos)**: Positive pairs (e.g., \( S_{1,1'}, S_{2,2'} \)) — These are the pairs that the loss function will maximize.
- **Black (Neg)**: Negative pairs (e.g., \( S_{1,2}, S_{1,3} \)) — These are pairs with different images, which the loss function will minimize.

### 3. Temperature Scaling

The similarities in the matrix \( S \) are divided by a temperature parameter \( \tau \), which sharpens or softens the contrastive learning objective. Lower temperatures make the model more sensitive to small differences in similarity.

\[
S_{i,j} = \frac{\text{cosine\_similarity}(z_i, z_j)}{\tau}
\]

### 4. Constructing Logits and Labels

For each sample \( z_i \):
- **Logits** are constructed by combining:
  - The positive similarity \( S_{i,j} \) (where \( j \) is the index of the positive pair).
  - All negative similarities \( S_{i,k} \) (where \( k \neq i \) and \( k \neq j \)) in that row.

The logits for each sample \( z_i \) look like:

\[
\text{logits}_i = \left[ \frac{S_{i,j}}{\tau}, \frac{S_{i,1}}{\tau}, \frac{S_{i,2}}{\tau}, \dots, \frac{S_{i,2N}}{\tau} \right]
\]

The label vector for the logits assigns a label `0` to the positive pair (target) and `1` for all negatives.

### 5. Cross-Entropy Loss

The NT-Xent loss applies cross-entropy loss to maximize the similarity between positive pairs and minimize the similarity between negative pairs. The formula for the NT-Xent loss for a positive pair \( (i, j) \) is:

\[
\ell_{i,j} = - \log \frac{\exp(S_{i,j} / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(S_{i,k} / \tau)}
\]

where:
- \( S_{i,j} \) is the positive pair similarity.
- \( S_{i,k} \) are the similarities with all other samples in the batch.
- \( \mathbb{1}_{[k \neq i]} \) is an indicator function that excludes self-similarities.

### 6. Averaging the Loss

The NT-Xent loss is averaged over all positive pairs in the batch to get the final loss, which is then used to update the model parameters.

## Summary of the NT-Xent Loss Process

1. **Data Augmentation**: Generate two augmented views of each image in the batch to create positive pairs.
2. **Compute Similarities**: Calculate pairwise cosine similarities to form the similarity matrix \( S \).
3

. **Temperature Scaling**: Scale similarities by a temperature parameter \( \tau \).
4. **Construct Logits**: For each sample, combine the positive pair similarity with all negative similarities.
5. **Cross-Entropy Loss**: Apply cross-entropy loss to maximize positive similarities and minimize negative similarities.
6. **Average the Loss**: Average the loss over all samples in the batch to obtain the final NT-Xent loss.

This loss function helps the model learn to make positive pairs more similar than any negative pairs in the batch, effectively learning useful representations for downstream tasks.
