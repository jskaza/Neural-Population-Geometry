#!/usr/bin/env python3
"""
Circular Manifold Simulation

This script simulates a neural network learning to predict the orientation of grating stimuli.
It then analyzes the geometry and topology of the network's internal representations to
uncover a circular manifold that represents the 360-degree orientation space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42) if torch.cuda.is_available() else None

# Configuration
class Config:
    # Data generation
    num_samples = 10000
    image_size = 28
    min_angle = 0
    max_angle = 360
    
    # Training
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 30
    
    # Model
    hidden_dim = 128
    latent_dim = 32  # Dimension of the layer we'll analyze
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Visualization
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

config = Config()

# Function to generate a grating stimulus with a given orientation
def generate_grating(angle, size=config.image_size, frequency=5):
    """Generate a sinusoidal grating with the specified orientation angle."""
    angle_rad = np.deg2rad(angle)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Rotate coordinates
    x_rot = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)
    
    # Create grating
    grating = np.sin(2 * np.pi * frequency * x_rot)
    
    # Normalize to [0, 1]
    grating = (grating + 1) / 2
    
    return grating

# Generate dataset
def generate_dataset(num_samples=config.num_samples):
    """Generate a dataset of grating stimuli with random orientations."""
    X = np.zeros((num_samples, 1, config.image_size, config.image_size))
    y_angle = np.zeros(num_samples)
    
    # Generate samples with uniform distribution of angles
    angles = np.random.uniform(config.min_angle, config.max_angle, num_samples)
    
    for i, angle in enumerate(angles):
        X[i, 0] = generate_grating(angle)
        y_angle[i] = angle
    
    # Convert angles to radians and then to sine and cosine components
    # This handles the circular nature of angles (360° = 0°)
    y_sin = np.sin(np.deg2rad(y_angle))
    y_cos = np.cos(np.deg2rad(y_angle))
    y = np.column_stack((y_sin, y_cos))
    
    return X, y, y_angle

# Neural Network Model
class GratingOrientationNet(nn.Module):
    def __init__(self):
        super(GratingOrientationNet, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Calculate the size after convolutions and pooling
        conv_output_size = config.image_size // 4  # After two 2x2 max pooling operations
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * conv_output_size * conv_output_size, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.ReLU(),
        )
        
        # Output layer (predicts sine and cosine of the angle)
        self.output_layer = nn.Linear(config.latent_dim, 2)
        
    def forward(self, x):
        x = self.conv_layers(x)
        latent = self.fc_layers(x)
        output = self.output_layer(latent)
        return output, latent

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=config.num_epochs):
    """Train the neural network model."""
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            
            # Forward pass
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return losses

# Function to evaluate the model
def evaluate_model(model, test_loader):
    """Evaluate the model and calculate the angular error."""
    model.eval()
    total_angular_error = 0.0
    
    with torch.no_grad():
        for inputs, targets, angles in test_loader:
            inputs = inputs.to(config.device)
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Convert predictions (sin, cos) back to angles
            pred_sin = outputs[:, 0].cpu().numpy()
            pred_cos = outputs[:, 1].cpu().numpy()
            pred_angles = np.rad2deg(np.arctan2(pred_sin, pred_cos)) % 360
            
            # Calculate angular error (considering the circular nature)
            errors = np.abs(pred_angles - angles.numpy())
            errors = np.minimum(errors, 360 - errors)
            total_angular_error += np.sum(errors)
    
    avg_angular_error = total_angular_error / len(test_loader.dataset)
    return avg_angular_error

# Function to extract latent representations
def extract_latent_representations(model, data_loader):
    """Extract latent representations from the model for all inputs."""
    model.eval()
    latent_reps = []
    angles = []
    
    with torch.no_grad():
        for inputs, _, angle_vals in data_loader:
            inputs = inputs.to(config.device)
            _, latent = model(inputs)
            latent_reps.append(latent.cpu().numpy())
            angles.append(angle_vals.numpy())
    
    return np.vstack(latent_reps), np.concatenate(angles)

# Function to visualize the latent space
def visualize_latent_space(latent_reps, angles, method='pca', filename='latent_space.png'):
    """Visualize the latent space using dimensionality reduction."""
    plt.figure(figsize=(10, 8))
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced_data = reducer.fit_transform(latent_reps)
        title = 'PCA of Latent Space'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        reduced_data = reducer.fit_transform(latent_reps)
        title = 't-SNE of Latent Space'
    elif method == 'umap':
        reducer = umap.UMAP(random_state=42)
        reduced_data = reducer.fit_transform(latent_reps)
        title = 'UMAP of Latent Space'
    
    # Create a scatter plot colored by angle
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                         c=angles, cmap='hsv', alpha=0.7, s=10)
    
    plt.colorbar(scatter, label='Angle (degrees)')
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, f"{method}_{filename}"))
    plt.close()

# Function to visualize sample grating stimuli
def visualize_gratings(angles=None):
    """Visualize sample grating stimuli at different orientations."""
    if angles is None:
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    n = len(angles)
    fig, axes = plt.subplots(1, n, figsize=(n*2, 2))
    
    for i, angle in enumerate(angles):
        grating = generate_grating(angle)
        axes[i].imshow(grating, cmap='gray')
        axes[i].set_title(f"{angle}°")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'grating_samples.png'))
    plt.close()

# Function to visualize the training loss
def plot_loss(losses):
    """Plot the training loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(config.results_dir, 'training_loss.png'))
    plt.close()

# Function to visualize the circular structure in 3D
def visualize_3d_manifold(latent_reps, angles, filename='3d_manifold.png'):
    """Visualize the first 3 principal components of the latent space."""
    # Apply PCA to get the first 3 principal components
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(latent_reps)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a scatter plot colored by angle
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                        c=angles, cmap='hsv', alpha=0.7, s=10)
    
    plt.colorbar(scatter, label='Angle (degrees)')
    ax.set_title('3D PCA of Latent Space')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, filename))
    plt.close()

# Main function
def main():
    print("Generating dataset...")
    X, y, angles = generate_dataset()
    
    # Split into train and test sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    angles_train, angles_test = angles[:split_idx], angles[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    angles_train_tensor = torch.FloatTensor(angles_train)
    angles_test_tensor = torch.FloatTensor(angles_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, angles_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, angles_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Visualize sample gratings
    print("Visualizing sample gratings...")
    visualize_gratings()
    
    # Initialize the model
    print("Initializing model...")
    model = GratingOrientationNet().to(config.device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Train the model
    print("Training model...")
    losses = train_model(model, train_loader, criterion, optimizer)
    
    # Plot the training loss
    plot_loss(losses)
    
    # Evaluate the model
    print("Evaluating model...")
    avg_angular_error = evaluate_model(model, test_loader)
    print(f"Average Angular Error: {avg_angular_error:.2f} degrees")
    
    # Extract latent representations
    print("Extracting latent representations...")
    latent_reps, latent_angles = extract_latent_representations(model, test_loader)
    
    # Visualize the latent space
    print("Visualizing latent space...")
    visualize_latent_space(latent_reps, latent_angles, method='pca')
    visualize_latent_space(latent_reps, latent_angles, method='tsne')
    visualize_latent_space(latent_reps, latent_angles, method='umap')
    visualize_3d_manifold(latent_reps, latent_angles)
    
    # Additional analysis: Fit a circle to the first 2 PCs
    print("Performing additional analysis...")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(latent_reps)
    
    # Plot the data colored by angle and fit a circle
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=latent_angles, cmap='hsv', alpha=0.7, s=10)
    
    # Calculate the center and radius of the best-fit circle
    center_x = np.mean(pca_data[:, 0])
    center_y = np.mean(pca_data[:, 1])
    radius = np.mean(np.sqrt((pca_data[:, 0] - center_x)**2 + (pca_data[:, 1] - center_y)**2))
    
    # Plot the circle
    circle = plt.Circle((center_x, center_y), radius, fill=False, color='black', linestyle='--', linewidth=2)
    plt.gca().add_patch(circle)
    
    plt.colorbar(scatter, label='Angle (degrees)')
    plt.title('PCA with Fitted Circle')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'pca_fitted_circle.png'))
    plt.close()
    
    print("Simulation complete! Results saved in the 'results' directory.")

if __name__ == "__main__":
    main()
