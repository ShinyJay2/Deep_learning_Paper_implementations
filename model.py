import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

## Transformer has Encoder - Decoder structure. 
# It is known that Encoder-only structure is better at understanding,
# while Decoder-only structure is better at generation.

### Transformer Encoder structure

# Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two 
# sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position- 2 wise 
# fully connected feed-forward network. We employ a residual connection [10] around each of the two sub-layers, 
# followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), 
# where Sublayer(x) is the function implemented by the sub-layer itself. 
# To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, 
# produce outputs of dimension dmodel = 512.

# Positional Embedding -> Multi-head Attention -> Add & Norm -> Feed Forward -> Add & Norm

# So what components(pytorch classes) do we need?

# 1. Multi-head self-attention mechanism
    # 1.1 Scaled Dot-product Attention (in order for multi-head use)
# 2. Position-2 wise fully-connected feed forward nn
# 3. Positional Encoding

## Section 3.2.1
class ScaledDotProductAttention(nn.Module):

    # Remember Attention being softmax(QK.T / sqrt(d_k)) V  ?
    # This is scaled attention so we need a parameter d_k to keep track of the dimension of K (Key matrix)

    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    # Here comes the famous Q K V
    # We also need mask for (Encoder: Masking Padding token / Decoder: Masking future token (to prevent cheating when training))
    # mask will be provided as tensors in train.py
    def forward(self, Q, K, V, mask=None):

        # @ being shrorthand for torch.matmul
        # permute() swaps dimensions, so we swapped 2 and 3 for matrix multiplication
        # Q : (batch_size, num_heads, seq_len_q, d_k)
        # K.transpose : (batch_size, num_heads, d_k, seq_len_k)
        # attention_score: (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_score = (Q @ K.permute(0, 1, 3, 2)) / math.sqrt(self.d_k)

        # Detailed explanation of Q, K, V
        # Q, K, V are matrices, which are outputs from W_q(x), W_k(x), W_v(x) linear projections.
        # Each row of Q in the matrix represents a token (word) from the input sequence (sentence).

        # x is the input sequence, the result of word embedding.
        # x would have the shape of: (batch_size, seq_len, d_model):
        # Q, K, V would have the shape of: (batch_size, seq_len, d_model)
        # batch_size: The number of sequences (sentences) in a batch.
        # seq_len: The number of tokens (e.g., words) in each sequence.
        # d_model: The embedding dimension. Dimensionality of the vector used to represent each token.
        # More dimensions mean a more diverse, informative representation of words (richer, complex relationships).

        # If we apply multi-head attention, we split the representation across multiple heads:
        # Q, K, V would have the shape of: (batch_size, num_heads, seq_len, d_k)
        #   where:
        #       batch_size: The number of sequences (sentences) in a batch.
        #       num_heads: The number of attention heads (splitting the representation into subspaces).
        #       seq_len: The number of tokens in each sequence.
        #       d_k: The dimension of the query/key vector for each head, d_model // num_heads

        # Each head focuses on a different part of the input, 
        # allowing the model to capture different aspects of the relationships in the sequence.

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float('-inf'))


        # attention_score would have the shape of : (batch_size, num_heads, seq_len_q, seq_len_k)
        # We need only the last dimension because this dimension contains the attention scores for each query
        # So, dim = -1
        attention_probability = F.softmax(attention_score, dim= -1)
        attention_output = torch.matmul(attention_probability, V)

        return attention_output


# Section 3.2.2
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Let's define attention too.
        self.attention = ScaledDotProductAttention(self.d_k)

        # We split the key_vector, value_vector into multiple-heads, so dims must be defined
        # However, we do linear projection 3 times to make Q, K, V. We are projecting input sequence x.

        ## Important! The process of MHA:
            # x, x, x -> Linear, Linear, Linear -> ScaledDotProductAttention -> concat -> Linear (concated one goes into here)

        # The first 3 Linears
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # The last 1 Linear (concated one goes here)
        self.W_o = nn.Linear(d_model, d_model)

        # input_sequence x has shape of 
    def forward(self, x_Q, x_K, x_V, mask=None):

        # We want nout batch_size, let's fetch it from input sequence x. 
        # The shape of x = (batch_size, seq_len, d_model), so let's fetch the 1st dimension of x
        batch_size = x_Q.size(0)  # Returns the size of the first dimension of x_Q

        # Let's make Q, K, V !!
        # 1. Linear projection by putting x_Q into the first 3 layers (self.W_q, self.W_k, self.W_v)
        # 2. Split dimensions into num_heads
        ## Q, K, V : (batch_size, seq_len, d_model) -> 
        Q = self.W_q(x_Q).reshape(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.W_k(x_K).reshape(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.W_v(x_V).reshape(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        # Why do we permute the dimensions?
        # If the dimensions were (batch_size, seq_len, num_heads, d_k), it would complicate the process 
        # because the num_heads dimension would not be easily accessible for parallel operations.

        # Apply attention
        attention_output = self.attention(Q, K, V, mask)

        # Concat multi-heads
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        attention_output = self.W_o(attention_output)
        #  .contiguous() ensures that the tensor is stored in a contiguous block of memory (required before .view() in some cases)

        return attention_output



# Section 3.3 
# Purpose: The FFN introduces non-linearity and allows for complex transformations of the input embeddings at each position independently.
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    # Before FFN: 
        # (batch_size, seq_len, d_model)
    # After 1st convolution:
        # (batch_size, seq_len, d_ff)
    # After 2nd convolution:
        # (batch_size, seq_len, d_model)

    ## The authors say: "Another way of describing this is as two convolutions with kernel size 1."
    ### Why? because 1x1 convolution is used for changing dimensions (sometimes for dimensionality reduction)


# Section 3.5 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()

        # We first initialize with a tensor full of zeros
        # Shape would look something like: (5000, 512)
        pe = torch.zeros(max_len, d_model)

        # In the paper, there is "pos", which is the position.
            # it would look like this : [0, 1, 2, ..., 4999]
                # The shapes do not match with the pe, so add an extra dimension
                # Shape would look something like: (5000, 1)
        pos = torch.arange(0, max_len).unsqueeze(1).float()


        ## PE(pos,2i) = sin ( pos / 10000^(2i/dmodel) )  --for even index
        ## PE(pos,2i+1) = cos ( pos / 10000^(2i/dmodel) )  --for odd index
            # i is for index!! so we need to keep using torch.arange() to use index numbers

        # Let's define the denomiator part
            # since d_model is the dimension of the embedding vector, we need this.
            # we need 2i, the index of the embedding vector, so use torch.arange untill the embedding vector dimension
        denominator = 1 / 10000 ** (torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(pos * denominator)  # --for even index
        pe[:, 1::2] = torch.cos(pos * denominator)  # --for odd index

        # our input sequence (sentence) x has shape of (batch_size, seq_len, d_model)
        # our pe has shape of (max_len, d_model)    -- seq_len is upper-bounded by max_len
            # So, for broadcasting purposes, we add an extra dimension in front (so it can easily be added across different batch sizes)
        pe = pe.unsqueeze(0)
        # Somekind of buffer
        self.register_buffer('pe', pe)

        # The result of the positional encoding would look like:
            # suppose max_len = 5 , d_model = 4
                # tensor([[ 0.0000,  1.0000,  0.0000,  1.0000],
                        # [ 0.0001,  0.9999,  0.0001,  0.9999],
                        # [ 0.0002,  0.9998,  0.0002,  0.9998],
                        # [ 0.0003,  0.9997,  0.0003,  0.9997],
                        # [ 0.0004,  0.9996,  0.0004,  0.9996]])

    def forward(self, x):

        # x = (batch_size, seq_len, d_model)
        seq_len = x.size(1) # selects the 1th dimension (python starts from 0)
        # pe = (1, max_len, d_model) 
            # since we .unsqueeze(0)
        # pe[:, :seq_len] 
            # the first : selects all of the first dimension. 
            # :seq_len slices the max_len "until the seq_len". Also it automatically selects all of the 3rd dimension, which is d_model
        positional_encoding = self.pe[:, :seq_len].to(x.device)  # Let's define the device in train.py

        x = x + positional_encoding
        return x


# Now the whole "1" Encoder layer  -Transformer consists of several encoder layers ("Nx" in the paper)

class EncoderLayer(nn.Module):

    # What params do we need? 
        # d_model, num_heads, d_ff
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Define layers!! 
            # MultiHeadAttention -> Add&Norm -> FeedForward -> Add&Norm
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm_after_attention = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)


        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm_after_feed_forward = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, x_mask):

        attention_output = self.attention(x, x, x, x_mask)
        x = x + attention_output # Residual connection (x represent a bypass)
        # ALREADY FORGOTTEN? you only need to write x inside of LayerNorm
        x = self.norm_after_attention(x)

        feed_forward_output = self.feed_forward(x)
        x = x + feed_forward_output # Residual connection
        x = self.norm_after_feed_forward(x)

        return x


# Now the whole "1" Decoder layer  -Transformer consists of several decoder layers ("Nx" in the paper)

## Decoder architecture explanation:

# 0. Output -> Output Embedding -> Positional Encoding
    # 1. Masked MultiHeadAttention / Add&Norm
    # 2. Cross-MultiHeadAttention / Add&Norm
    # 3. PointwiseFeedForward / Add&Norm
# 4. Linear / Softmax / Output probability

# 0. and 4. is not the decoder architecture, but the workflow of the decoder.
# Decoder is trained in pararell with the Encoder
    # sequence-to-sequence Transformer model (like in translation tasks), 
    # the training process involves passing the source language (e.g., Korean) into the encoder 
    # and the target language (e.g., English) into the decoder in parallel.

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.masked_attention = MultiHeadAttention(d_model, num_heads)
        self.norm_after_masked_attention = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.norm_after_cross_attention = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm_after_ffn = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    # Okay, we need to define our forward(). The Decoder processes the "target input sequence x".
    # First, we do masked multi head attention for "not attending to the future (not yet known) values."
    # Second, we perform cross multi head attention with the ouputs of the Encoder, 
        # So we are doing attention on Q, K from Encdoer, V frm the Deocder
    # Add residual_conndection

    # What parameters do we need? 
        # 1. target_x : since we are dealing with the "translation target input sequence(sentence)"
        # 2. encoder_output : since we have to perform cross-multi head attention
        # 3. encoder_mask , decoder_mask : We have padding mask for encdoer, and future token mask in the decoder_mask
    def forward(self, target_x, encoder_output, encoder_mask, decoder_mask):

        attention_output = self.masked_attention(target_x, target_x, target_x, decoder_mask)
         # Used decoder_mask to prevent attending to future target

        # Add residual and attention_output and Normalize the whole thing.
        target_x = target_x + attention_output
        target_x = self.norm_after_masked_attention(target_x)

        # Perform cross-attention with encoder_output 
        ## Important! In MultiHeadAttention, the matrix order is V, K, Q
        ### Represents the sequence to which we want to find relevant context. 
        ### In the decoder's cross-attention, the target_x (target sequence representation) should act as the query.
        cross_attention_output = self.cross_attention(target_x, encoder_output, encoder_output, encoder_mask)
        # Why encoder_mask here? Because the decoder Q attends to the "encoder K, V", so we need "encoder_mask"

        # Add residual
        target_x = target_x + cross_attention_output
        target_x = self.norm_after_cross_attention(target_x)

        ffn_output = self.feed_forward(target_x)
        
        # Add residual
        target_x = target_x + ffn_output
        target_x = self.norm_after_ffn(target_x)

        return target_x
    

# We created 1 EncoderLayer, now is the time to stack'em up.

class Encoder(nn.Module):
    def __init__(self, input_x_len, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Important!!!
        # We have input embeddings in Encoder, and output embeddings in Decoder.
        # This is where we convert our shape of input sequence (or sentence)
        # Before input embedding: x -> (batch_size, seq_len)
        # After input embedding: x -> (batch_size, seq_len, d_model)

        ## We defined d_model as : The embedding dimension, which is the size of the dense vector representation for each token.
        ## Let's not be lost in swirling notations!!

        # Let's think about what do we need here, and what parameters do we need.

        # We need Input Embedding
        self.embedding = nn.Embedding(input_x_len, d_model)

        # We need Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Stack some layers
        # Can use nn.ModuleList(), allows you to store multiple copies of these layers in an organized manner.
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, encoder_mask)
        
        x = self.norm(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self, target_x_len, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(target_x_len, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_x, encoder_output, decoder_mask, encoder_mask):
        target_x = self.embedding(target_x) * math.sqrt(self.d_model)
        target_x = self.pos_encoding(target_x)

        for layer in self.layers:
            target_x = layer(target_x, encoder_output, encoder_mask, decoder_mask)
        
        target_x = self.norm(target_x)

        return target_x
    


class Transformer(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, d_model=256, num_layers=4, num_heads=4, d_ff=1024, max_len=1000):
        super().__init__()
        self.encoder = Encoder(input_vocab_size, d_model, num_layers, num_heads, d_ff, max_len)
        self.decoder = Decoder(target_vocab_size, d_model, num_layers, num_heads, d_ff, max_len)
        self.output_layer = nn.Linear(d_model, target_vocab_size)  # Output projection layer to convert to vocabulary probabilities (Section 3.4)

    def forward(self, input_x, target_x, encoder_mask, decoder_mask):
        encoder_output = self.encoder(input_x, encoder_mask)  # Encode the input sequence (Section 3.1)
        output = self.decoder(target_x, encoder_output, decoder_mask, encoder_mask)  # Decode using encoder outputs (Section 3.1)
        output = self.output_layer(output)  # Final linear layer to project to vocabulary size
        return output
    

# To fit fit the 8GB memory of your M2 MacBook Air by making the following changes:
    # d_model reduced to 256
    # num_layers reduced to 4
    # num_heads reduced to 4
    # d_ff reduced to 1024
    # max_len reduced to 1000

def initialize_parameters(model):
    """
    Initialize the parameters of the model.
    - Weights are initialized using Xavier Uniform Initialization.
    - Biases are initialized to zero.
    """
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
            print(f"Initialized {name} with Xavier Uniform")
        elif 'bias' in name:
            nn.init.zeros_(param)
            print(f"Initialized {name} with zeros")