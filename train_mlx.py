import os
import time
import math
import pickle
from contextlib import nullcontext
from mlx.utils import tree_flatten, tree_map
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from functools import partial

# Import the Llama model from your provided script
from mlx_model import ModelArgs, Model
import wandb 


# -----------------------------------------------------------------------------
# Configuration values, analogous to the nanoGPT configuration
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume'
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 8
block_size = 512
data_dir = os.path.join('data', dataset)



# 'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params

iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")


vocab_size = meta_vocab_size
learning_rate = 6e-4
max_iters = 100
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# -----------------------------------------------------------------------------
# Set up directories and initialize variables
os.makedirs(out_dir, exist_ok=True)
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"Tokens per iteration: {tokens_per_iter:,}")


# Update Model configuration to match GPT-2 parameters (124M params)
dim = 768           # Set `dim` to match `n_embd` of GPT-2
n_layers = 12       # Set `n_layers` to 12, matching GPT-2
head_dim = 64       # Keep `head_dim` unchanged; each head's dimension
hidden_dim = 4 * dim  # hardcoded in MLX MLP 
n_heads = 12        # Number of attention heads, matching GPT-2
norm_eps = 1e-5     # Epsilon for layer normalization



model_args = ModelArgs(
    model_type="gpt2",
    n_ctx=block_size,
    n_embd=dim,
    n_head=n_heads,
    n_layer=n_layers,
    n_positions=block_size, 
    layer_norm_epsilon=norm_eps, # Epsilon value for layer normalization
    vocab_size=vocab_size,       # Size of the vocabulary
)


# Initialize the Model class using the same ModelArgs instance
model = Model(model_args)
model.apply(lambda x: x.astype(mx.bfloat16))
# Assuming `model` is your MLX model instance
mx.eval(model.parameters())
nparams = sum(
    x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
)
print(f"Training a transformer with {nparams / 1e6:.3f} M parameters")


def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction="mean")

optimizer = optim.AdamW(
    learning_rate=learning_rate, 
    weight_decay=weight_decay, 
    betas=[beta1, beta2]  # Add the betas parameter
)


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

@partial(mx.compile, inputs=model.state, outputs=model.state)
def step(X, y):
    # Compute loss and gradients without updating the model
    loss, grads = loss_and_grad_fn(model, X, y)
    return loss, grads

# Set up a scheduler function for learning rate decay
def get_lr(it):
    if it < 2000:  # Warmup phase
        return learning_rate * it / 2000
    if it > max_iters:  # Minimum learning rate phase
        return learning_rate / 10
    # Cosine decay
    decay_ratio = (it - 2000) / (max_iters - 2000)
    return learning_rate / 10 + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (learning_rate - learning_rate / 10)


def get_batch(split):
    # Load data from the binary file depending on the split (train/val)
    data_dir = os.path.join('data', dataset)
    data_file = os.path.join(data_dir, f'{split}.bin')
    
    # Open the memmapped data file
    data = np.memmap(data_file, dtype=np.uint16, mode='r')
    
    # Randomly select indices to create batches
    idx = np.random.randint(0, len(data) - block_size - 1, batch_size)
    
    # Convert slices of `data` to numpy arrays and then to `mlx.array`
    x = mx.array(np.array([data[i:i + block_size].astype(np.int64) for i in idx]))
    y = mx.array(np.array([data[i + 1:i + 1 + block_size].astype(np.int64) for i in idx]))
    
    return x, y

# 
def estimate_loss():
    """
    Compute and return the average loss for both the train and validation datasets.

    This function evaluates the model over a specified number of evaluation iterations (eval_iters)
    and returns the average loss for each dataset split (train and val).
    """
    # Dictionary to store average losses for train and validation splits
    losses = {'train': [], 'val': []}

    # Loop through both training and validation splits
    for split in ['train', 'val']:
        total_loss = 0.0  # Initialize total loss for the current split

        # Iterate through eval_iters to compute the loss on multiple batches
        for _ in range(eval_iters):
            # Get a batch of data (X: input, Y: target labels)
            X, Y = get_batch(split)

            # Perform a forward pass through the model to get predictions (logits)
            logits = model(X)

            # Calculate the loss using cross-entropy loss
            loss = nn.losses.cross_entropy(logits, Y, reduction="mean")

            # Accumulate the total loss for the current split
            total_loss += loss.item()

        # Calculate the average loss for the current split
        average_loss = total_loss / eval_iters
        losses[split] = average_loss

    return losses




def train(max_its):
    # Initialize variables for training
    iter_num = 0
    best_val_loss = float('inf')
    running_loss = 0.0

    # Start the training loop
    while iter_num < max_its:
        print(iter_num, max_iters)
        # Adjust the learning rate based on the current iteration
        lr = get_lr(iter_num)
        optimizer.learning_rate = lr  # Update optimizer's learning rate

        # Fetch a batch of training data
        X, Y = get_batch('train')

        # Perform forward and backward pass to get loss and gradients
        loss, grads = step(X, Y)

        # Update model parameters using optimizer
        optimizer.update(model, grads)

        # Update running loss for logging
        running_loss += loss.item()

        # Log training progress
        if (iter_num + 1) % log_interval == 0:
            avg_running_loss = running_loss / log_interval
            print(f"Iter {iter_num + 1}: loss = {avg_running_loss:.4f}, lr = {lr:.6f}")
            running_loss = 0.0
        if iter_num % log_interval == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            wandb.log({
                "iter": iter_num,
                "train/loss": loss.item() * gradient_accumulation_steps,
                "lr": lr,
            
            })
        # Evaluate the model periodically
        if (iter_num + 1) % eval_interval == 0:
            # ommitted for fairest comparison 
            pass  # Replace with evaluation logic

        # Increment iteration counter
        iter_num += 1

# -----------------------------------------------------------------------------
# Main function to start training or evaluation
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a Llama model with MLX.")
    parser.add_argument("--dataset", type=str, default="shakespeare_char", help="Name of the dataset")
    parser.add_argument("--out-dir", type=str, default="out", help="Directory to save checkpoints and logs")
    parser.add_argument("--max-iters", type=int, default=600000, help="Maximum number of iterations to train")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate the model, without training")
    args = parser.parse_args()

    # Update configuration with command line arguments
    dataset = args.dataset
    out_dir = args.out_dir
    max_iters = 100
    eval_only = args.eval_only

    # Create necessary directories
    os.makedirs(out_dir, exist_ok=True)

    # Either train or evaluate the model
    if eval_only:
        print("Running evaluation only...")
        model.eval()
        losses = estimate_loss()
        print(f"Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}")
    else:
        print("Starting training...")
        wandb.init(project="mlx_vs_torch")

        train(max_iters)

