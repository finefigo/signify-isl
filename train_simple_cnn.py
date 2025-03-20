import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# Define the CNN model with updated architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Custom Dataset class
class GestureDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_data(data_dir='training_data'):
    """Load and preprocess the image data."""
    images = []
    labels = []
    gesture_mapping = {}
    
    # Get gesture directories
    gesture_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found {len(gesture_dirs)} gesture classes: {gesture_dirs}")
    
    # Create gesture mapping
    for i, gesture in enumerate(gesture_dirs):
        gesture_mapping[str(i)] = gesture
    
    # Load images
    for label, gesture in enumerate(gesture_dirs):
        gesture_dir = os.path.join(data_dir, gesture)
        print(f"Loading images for gesture '{gesture}'...")
        
        image_files = [f for f in os.listdir(gesture_dir) if f.endswith('.jpg')]
        print(f"Found {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(gesture_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))
                # Normalize and transpose for PyTorch (N, C, H, W)
                img = img.transpose(2, 0, 1) / 255.0
                images.append(img)
                labels.append(label)
    
    return np.array(images), np.array(labels), gesture_mapping

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the model and return training history."""
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
    
    return train_losses, val_losses, train_accs, val_accs

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot and save training history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model/training_history.png')
    plt.close()

def export_model(model, gesture_mapping):
    """Export the model for use in JavaScript."""
    os.makedirs('model', exist_ok=True)
    
    # Save gesture mapping
    with open('model/gesture_mapping.json', 'w') as f:
        json.dump(gesture_mapping, f, indent=4)
    
    # Save model state
    torch.save(model.state_dict(), 'model/model.pth')
    
    # Create a simplified model.json for JavaScript
    model_config = {
        "format": "PyTorch",
        "modelTopology": {
            "layers": [
                {"type": "conv2d", "filters": 32, "kernelSize": 3},
                {"type": "maxpool2d", "size": 2},
                {"type": "conv2d", "filters": 64, "kernelSize": 3},
                {"type": "maxpool2d", "size": 2},
                {"type": "conv2d", "filters": 128, "kernelSize": 3},
                {"type": "maxpool2d", "size": 2},
                {"type": "flatten"},
                {"type": "dense", "units": 512},
                {"type": "dense", "units": len(gesture_mapping)}
            ]
        }
    }
    
    with open('model/model.json', 'w') as f:
        json.dump(model_config, f, indent=4)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    images, labels, gesture_mapping = load_data()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets and dataloaders
    train_dataset = GestureDataset(X_train, y_train)
    val_dataset = GestureDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create model
    num_classes = len(gesture_mapping)
    model = SimpleCNN(num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print('Training model...')
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, num_epochs=25,
        device=device
    )
    
    # Plot training history
    plot_training_history(*history)
    
    # Export model
    print('Exporting model...')
    export_model(model, gesture_mapping)
    
    print('Training complete!')

if __name__ == '__main__':
    main() 