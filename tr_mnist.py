# import torch
# import torch.utils.data as data
# import torchvision
# import torchvision.transforms as transforms
# import mlx.core as mx
# import mlx.nn as nn
# import mlx.optimizers as optim

# # Define a basic MLP model using MLX
# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)

#     def __call__(self, x):
#         x = nn.relu(self.fc1(x))
#         x = nn.relu(self.fc2(x))
#         return self.fc3(x)

# # Define a simple training function
# def train(model, train_loader, optimizer, loss_fn, num_epochs):
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for batch_idx, (inputs, targets) in enumerate(train_loader):
#             # Convert to MLX arrays
#             inputs, targets = mx.array(inputs), mx.array(targets)

#             # Flatten inputs for MLP
#             inputs = inputs.reshape([inputs.shape[0], -1])

#             # Forward pass
#             logits = model(inputs)

#             # Calculate loss
#             loss = loss_fn(logits, targets)

#             # Compute gradients and update weights
#             # optimizer.zero_grad()
#             grads = mx.grad(model, loss)
#             optimizer.update(model, grads)

#             # Track loss
#             running_loss += loss.item()
        
#         # Print average loss after each epoch
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# # Define a simple evaluation function
# def evaluate(model, test_loader, loss_fn):
#     total_correct = 0
#     total_samples = 0
#     total_loss = 0.0

#     for inputs, targets in test_loader:
#         # Convert to MLX arrays
#         inputs, targets = mx.array(inputs), mx.array(targets)

#         # Flatten inputs for MLP
#         inputs = inputs.reshape([inputs.shape[0], -1])

#         # Forward pass
#         logits = model(inputs)
#         loss = loss_fn(logits, targets)

#         # Calculate accuracy
#         predicted = mx.argmax(logits, axis=1)
#         total_correct += mx.sum(predicted == targets).item()
#         total_samples += targets.size
#         total_loss += loss.item()

#     # Calculate and print average loss and accuracy
#     avg_loss = total_loss / len(test_loader)
#     accuracy = total_correct / total_samples
#     print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# # Main function to run the training and evaluation
# def main():
#     # Hyperparameters
#     input_dim = 28 * 28  # MNIST images are 28x28
#     hidden_dim = 128
#     output_dim = 10  # 10 classes in MNIST
#     learning_rate = 0.01
#     num_epochs = 500
#     batch_size = 64

#     # Load MNIST dataset using PyTorch
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#     train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
#     test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

#     # Create data loaders
#     train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     # Initialize model, optimizer, and loss function
#     model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
#     optimizer = optim.SGD(learning_rate=learning_rate)
#     loss_fn = nn.losses.cross_entropy

#     # Train the model
#     train(model, train_loader, optimizer, loss_fn, num_epochs)

#     # Evaluate the model
#     evaluate(model, test_loader, loss_fn)

# # Run the main function
# if __name__ == "__main__":
#     main()

import argparse
import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from functools import partial

# Model: Simple MLP
class MLP(nn.Module):
    """A simple MLP."""
    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
        return self.layers[-1](x)


# Loss function
def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction="mean")

# Data loader with MLX-compatible batching
def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s: s + batch_size]
        yield X[ids], y[ids]

# Training function with debugging prints
def train(args):
    seed = 0
    num_layers = 2
    hidden_dim = 128  # Increase hidden size for better performance
    num_classes = 10
    batch_size = 64
    num_epochs = 10000
    learning_rate = 0.01

    np.random.seed(seed)

    # Load MNIST data using torchvision (convert to numpy for MLX compatibility)
    import torchvision.transforms as transforms
    import torch
    import torchvision

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Convert PyTorch datasets to NumPy arrays and then to MLX arrays
    train_images = mx.array(train_dataset.data.numpy().reshape(-1, 28*28).astype(np.float32) / 255.0)
    train_labels = mx.array(train_dataset.targets.numpy().astype(np.int32))
    test_images = mx.array(test_dataset.data.numpy().reshape(-1, 28*28).astype(np.float32) / 255.0)
    test_labels = mx.array(test_dataset.targets.numpy().astype(np.int32))

    # Initialize the model
    model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
    mx.eval(model.parameters())

    # Define optimizer
    optimizer = optim.SGD(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Compile the step function with model state
    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def step(X, y):
        # Debugging: Print the types of X and y
        print(f"X type: {type(X)}, X shape: {X.shape}")
        print(f"y type: {type(y)}, y shape: {y.shape}")

        # Calculate loss and gradients
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss

    # Compile the evaluation function
    @partial(mx.compile, inputs=model.state)
    def eval_fn(X, y):
        return mx.mean(mx.argmax(model(X), axis=1) == y)

    # Training loop with debugging
    for epoch in range(num_epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            step(X, y)
            mx.eval(model.state)  # Ensure the model state is evaluated
        accuracy = eval_fn(test_images, test_labels)
        toc = time.perf_counter()
        print(f"Epoch {epoch + 1}: Test accuracy {accuracy.item():.3f}, Time {toc - tic:.3f} (s)")

# Main function to run training or evaluation
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple MLP on MNIST with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="The dataset to use.",
    )
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    train(args)

