import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

# Check for device and print it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up data loading for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(model, train_loader, optimizer, criterion, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

# Initialize model, loss function, and optimizer
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Compile the model
model = torch.compile(model)

# Training and evaluation with float32 (default)
print("Training with float32...")
train_model(model, train_loader, optimizer, criterion)
print("Evaluating with float32...")
evaluate_model(model, test_loader)

# Change model to float16 (requires CUDA)
if device.type == 'cuda':
    print("Training with float16...")
    model.half()  # Convert model to float16
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
            layer.float()  # Keep batch norm layers in float32 for stability

    train_loader_float16 = [(inputs.half().to(device), labels.to(device)) for inputs, labels in train_loader]
    train_model(model, train_loader_float16, optimizer, criterion)
    evaluate_model(model, [(inputs.half().to(device), labels.to(device)) for inputs, labels in test_loader])

# Change model to bfloat16
print("Training with bfloat16...")
model.bfloat16()  # Convert model to bfloat16
train_loader_bfloat16 = [(inputs.bfloat16().to(device), labels.to(device)) for inputs, labels in train_loader]
train_model(model, train_loader_bfloat16, optimizer, criterion)
evaluate_model(model, [(inputs.bfloat16().to(device), labels.to(device)) for inputs, labels in test_loader])
