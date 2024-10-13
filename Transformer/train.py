# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
from datasets import load_dataset
from model import Transformer, initialize_parameters  # Ensure initialize_parameters is imported
import math
import time
from tqdm import tqdm  # For progress bars
from torch.utils.tensorboard import SummaryWriter  # For logging and visualization

# For custom label smoothing (if needed)
import torch.nn.functional as F

# Define LabelSmoothingLoss if PyTorch < 1.10
class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, vocab_size, ignore_index=0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')  # Using KL Divergence loss
        self.padding_idx = ignore_index
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.vocab_size = vocab_size

    def forward(self, x, target):
        """
        x: (N, C) where C = number of classes
        target: (N,) where each value is 0 <= targets[i] <= C-1
        """
        assert x.size(1) == self.vocab_size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        ignore = target == self.padding_idx
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist.masked_fill_(ignore.unsqueeze(1), 0)
        return self.criterion(F.log_softmax(x, dim=1), true_dist) / x.size(0)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
input_vocab_size = 10000
target_vocab_size = 10000
max_len = 1000
d_model = 256        # Updated to match model.py
num_layers = 4       # Updated to match model.py
num_heads = 4        # Updated to match model.py
d_ff = 1024          # Updated to match model.py
learning_rate = 1e-3  # Base learning rate before scaling
num_epochs = 50
batch_size = 4
grad_accum_steps = 4  # To simulate a larger batch size
dropout = 0.1        # Match dropout rate in model.py
max_grad_norm = 1.0  # For gradient clipping
patience = 5         # For early stopping
log_dir = "runs/transformer_experiment"  # TensorBoard log directory
warmup_steps = 4000  # As per original Transformer

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)

# Create model and move to device
model = Transformer(
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    max_len=max_len
).to(device)

# Initialize parameters
initialize_parameters(model)

# Loss and optimizer
# Choose whether to use built-in label smoothing or custom
use_builtin_label_smoothing = False  # Set to True if using PyTorch >= 1.10

if use_builtin_label_smoothing:
    label_smoothing = 0.1  # As per original Transformer
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
else:
    label_smoothing = 0.1  # As per original Transformer
    criterion = LabelSmoothingLoss(label_smoothing=label_smoothing, vocab_size=target_vocab_size, ignore_index=0)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

# Learning rate scheduler: Implementing the original Transformer schedule
def get_lr_lambda(current_step):
    if current_step == 0:
        return 1.0
    return (d_model ** -0.5) * min(current_step ** -0.5, current_step * (warmup_steps ** -1.5))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)

# Define tokenizers using basic tokenizers
src_tokenizer = lambda x: x.lower().split()
trg_tokenizer = lambda x: x.lower().split()

# Load the dataset using Hugging Face Datasets
print("Loading dataset...")
raw_dataset = load_dataset('wmt16', 'de-en', split='train[:1%]')  # Load only 1% of the data to fit 8GB RAM
print("Dataset loaded.")

# Split dataset into train and validation
train_valid_split = raw_dataset.train_test_split(test_size=0.1)
train_data = train_valid_split['train']
valid_data = train_valid_split['test']

# Build vocabulary
def build_vocab(data_iter, tokenizer, max_tokens=10000, specials=['<pad>', '<sos>', '<eos>', '<unk>']):
    counter = Counter()
    for data in data_iter:
        tokens = tokenizer(data['translation']['en'])
        counter.update(tokens)
    # Start indexing after special tokens
    vocab = {token: idx + len(specials) for idx, (token, _) in enumerate(counter.most_common(max_tokens))}
    for idx, special in enumerate(specials):
        vocab[special] = idx
    return vocab

print("Building source vocabulary...")
SRC_vocab = build_vocab(train_data, src_tokenizer, max_tokens=input_vocab_size)
SRC_vocab_default = SRC_vocab.get('<unk>', 0)

print("Building target vocabulary...")
TRG_vocab = build_vocab(train_data, trg_tokenizer, max_tokens=target_vocab_size)
TRG_vocab_default = TRG_vocab.get('<unk>', 0)

# Create inverse vocabularies (optional, for decoding)
# SRC_inv_vocab = {v: k for k, v in SRC_vocab.items()}
# TRG_inv_vocab = {v: k for k, v in TRG_vocab.items()}

# Create DataLoader
def collate_fn(batch):
    src_batch, trg_batch = [], []
    for data in batch:
        src_tokens = src_tokenizer(data['translation']['en'])
        trg_tokens = trg_tokenizer(data['translation']['de'])
        
        # Convert tokens to indices and add <sos> and <eos>
        src_indices = [SRC_vocab.get('<sos>')] + [SRC_vocab.get(token, SRC_vocab_default) for token in src_tokens] + [SRC_vocab.get('<eos>')]
        trg_indices = [TRG_vocab.get('<sos>')] + [TRG_vocab.get(token, TRG_vocab_default) for token in trg_tokens] + [TRG_vocab.get('<eos>')]
        
        src_batch.append(torch.tensor(src_indices, dtype=torch.long))
        trg_batch.append(torch.tensor(trg_indices, dtype=torch.long))
    
    # Pad sequences
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=SRC_vocab['<pad>'])
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=TRG_vocab['<pad>'])
    
    return src_batch, trg_batch

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Evaluation function
def evaluate(model, valid_loader, criterion, SRC_vocab, TRG_vocab, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_x, target_x in valid_loader:
            encoder_mask = (input_x != SRC_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
            decoder_mask = (target_x != TRG_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
            
            input_x = input_x.to(device)
            target_x = target_x.to(device)
            encoder_mask = encoder_mask.to(device)
            decoder_mask = decoder_mask.to(device)
            
            outputs = model(input_x, target_x, encoder_mask, decoder_mask)
            
            # Shift target for teacher forcing
            target_y = target_x[:, 1:].contiguous().view(-1)
            outputs = outputs[:, :-1, :].contiguous().view(-1, target_vocab_size)
            
            loss = criterion(outputs, target_y)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(valid_loader)
    model.train()
    return avg_loss

# Early Stopping initialization
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

print("Starting training...")
current_step = 0  # To track the number of optimization steps

for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    epoch_loss = 0.0
    optimizer.zero_grad()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    
    for batch_idx, (input_x, target_x) in enumerate(progress_bar):
        encoder_mask = (input_x != SRC_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
        decoder_mask = (target_x != TRG_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
        
        # Move data to device
        input_x = input_x.to(device)
        target_x = target_x.to(device)
        encoder_mask = encoder_mask.to(device)
        decoder_mask = decoder_mask.to(device)
        
        # Forward pass
        outputs = model(input_x, target_x, encoder_mask, decoder_mask)
        
        # Shift target for teacher forcing
        target_y = target_x[:, 1:].contiguous().view(-1)
        outputs = outputs[:, :-1, :].contiguous().view(-1, target_vocab_size)
        
        # Compute loss
        loss = criterion(outputs, target_y)
        loss = loss / grad_accum_steps  # Normalize loss for gradient accumulation
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            
            # Update learning rate scheduler
            scheduler.step(current_step)
            current_step += 1
        
        epoch_loss += loss.item() * grad_accum_steps  # Accumulate actual loss
        
        # Logging to TensorBoard
        global_step = (epoch - 1) * len(train_loader) + batch_idx
        writer.add_scalar('Loss/train', loss.item() * grad_accum_steps, global_step)
        
        # Update progress bar
        progress_bar.set_postfix({"Loss": f"{loss.item() * grad_accum_steps:.4f}"})
    
    # Handle remaining gradients if total steps not divisible by grad_accum_steps
    if len(train_loader) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        
        # Update learning rate scheduler
        scheduler.step(current_step)
        current_step += 1
    
    end_time = time.time()
    epoch_duration = end_time - start_time
    avg_epoch_loss = epoch_loss / len(train_loader)
    
    # Validation
    val_loss = evaluate(model, valid_loader, criterion, SRC_vocab, TRG_vocab, device)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    
    # Logging epoch metrics
    print(f"Epoch [{epoch}/{num_epochs}] completed in {epoch_duration:.2f} seconds. Average Training Loss: {avg_epoch_loss:.4f}. Validation Loss: {val_loss:.4f}")
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    
    # Early Stopping Check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_transformer_model.pth')
        print(f"Validation loss improved. Model saved.")
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            early_stop = True
            break

    if early_stop:
        break

print("Training complete.")

# Save the final model
torch.save(model.state_dict(), 'final_transformer_model.pth')
print("Final model saved as final_transformer_model.pth")

# Close the TensorBoard writer
writer.close()
