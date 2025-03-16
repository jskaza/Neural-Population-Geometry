#!/usr/bin/env python3
"""
Circular Manifold Simulation

This script simulates a neural network learning to predict the orientation of grating stimuli.
It then analyzes the geometry and topology of the network's internal representations to
uncover a circular manifold that represents the orientation space, accounting for the
180-degree symmetry of gratings (0° and 180° look identical, as do 90° and 270°).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import os
from tqdm import tqdm
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, MDS
import warnings
import pandas as pd
from scipy.optimize import curve_fit
from manifold_utils import analyze_pca, analyze_circular_regression, visualize_circular_colormap

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42) if torch.cuda.is_available() else None

# Configuration
class Config:
    # Data generation
    num_samples = 10000
    image_size = 28
    
    # Training
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 30
    
    # Model
    hidden_dim = 128
    latent_dim = None
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Visualization
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Neuron data file
    neuron_data_file = 'cell_properties.csv'

config = Config()

def load_neuron_orientations():
    """Load orientation values from neuron data"""
    # Load the data
    df = pd.read_csv(config.neuron_data_file)
    
    # Parse string arrays for orientations
    df['orientations'] = df['orientations'].apply(lambda s: [float(x) for x in s.strip('[]').split()])
    
    # Filter cells based on quality metrics (same as in neuron analysis)
    filtered_df = df[(df['osi'] >= 0.5) & (df['r_squared'] >= 0.6)]
    
    # Get orientations and set latent_dim based on neuron count
    orientations = np.array(filtered_df['orientations'].iloc[0])
    config.latent_dim = len(filtered_df)
    
    print(f"Latent dimension: {config.latent_dim}")
    print(f"Loaded {len(orientations)} orientations from neuron data")
    
    return orientations
        
# Function to generate a grating stimulus with a given orientation
def generate_grating(angle_degrees, size=config.image_size, frequency=10):
    """Generate a sinusoidal grating with the specified orientation angle in degrees.
    
    Note: Due to the nature of gratings, orientations that are 180° apart look identical.
    0° corresponds to a horizontal grating (bars running horizontally).
    90° corresponds to a vertical grating (bars running vertically).
    """
    angle_degrees = angle_degrees % 180
    angle_rad = np.deg2rad(angle_degrees)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Corrected rotation for proper orientation alignment
    x_rot = xx * np.cos(angle_rad) - yy * np.sin(angle_rad)
    
    # Create grating
    grating = np.sin(2 * np.pi * frequency * x_rot)
    
    # Normalize to [0, 1]
    grating = (grating + 1) / 2
    
    return grating

# Generate dataset
def generate_dataset(orientations):
    """Generate a dataset of grating stimuli with cos(2θ) and sin(2θ) labels."""
    num_orientations = len(orientations)
    samples_per_orientation = config.num_samples // num_orientations
    total_samples = samples_per_orientation * num_orientations

    X = np.zeros((total_samples, 1, config.image_size, config.image_size))
    y = np.zeros((total_samples, 2))  # [cos(2θ), sin(2θ)]
    o = np.zeros(total_samples,)
    sample_idx = 0
    for orientation in orientations:
        for _ in range(samples_per_orientation):
            jittered_angle = (orientation)
            X[sample_idx, 0] = generate_grating(jittered_angle)

            # Convert to radians and compute cos(2θ), sin(2θ)
            theta_rad = np.deg2rad(jittered_angle * 2) + np.random.normal(0, 0.1) # noise to simulate firing rate variability
            y[sample_idx, 0] = np.cos(theta_rad)  # cos(2θ)
            y[sample_idx, 1] = np.sin(theta_rad)  # sin(2θ)
            o[sample_idx] = jittered_angle
            sample_idx += 1

    # Shuffle dataset
    shuffle_idx = np.random.permutation(total_samples)
    X, y, o = X[shuffle_idx], y[shuffle_idx], o[shuffle_idx]
    
    return X, y, o

# Neural Network Model
class GratingOrientationNet(nn.Module):
    def __init__(self):
        super(GratingOrientationNet, self).__init__()
        self.latent_dim = config.latent_dim
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
            nn.Linear(config.hidden_dim, self.latent_dim),
            nn.ReLU(),
        )
        
        # Output layer (predicts sin and cos of 2*angle)
        self.output_layer = nn.Linear(self.latent_dim, 2)  # [cos(2θ), sin(2θ)]
        
    def forward(self, x):
        x = self.conv_layers(x)
        latent = self.fc_layers(x)
        raw_output = self.output_layer(latent)
        
        # Normalize the output to ensure it's on the unit circle
        norm = torch.sqrt(raw_output[:, 0]**2 + raw_output[:, 1]**2).unsqueeze(1)
        normalized_output = raw_output / (norm + 1e-8)
        
        return normalized_output, latent

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=config.num_epochs):
    """Train the CNN to predict cos(2θ) and sin(2θ)."""
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            outputs, _ = model(inputs)  # Predict cos(2θ), sin(2θ)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# Function to evaluate the model
def evaluate_model(model, test_loader):
    """Evaluate model performance using angular error."""
    model.eval()
    total_angular_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs = inputs.to(config.device)

            # Forward pass
            outputs, _ = model(inputs)
            outputs = outputs.cpu().numpy()  # Get predictions

            # Convert (cos(2θ), sin(2θ)) back to angles
            pred_angles_rad = np.arctan2(outputs[:, 1], outputs[:, 0]) / 2
            pred_angles_deg = np.rad2deg(pred_angles_rad) % 180  # Keep in [0, 180]

            # Convert ground truth back to angles
            true_angles_rad = np.arctan2(targets[:, 1].cpu().numpy(), targets[:, 0].cpu().numpy()) / 2
            true_angles_deg = np.rad2deg(true_angles_rad) % 180

            # Compute angular error
            angular_errors = np.abs(pred_angles_deg - true_angles_deg)
            angular_errors = np.minimum(angular_errors, 180 - angular_errors)

            total_angular_error += np.sum(angular_errors)
            total_samples += len(targets)

    return total_angular_error / total_samples  # Mean error in degrees

# Function to extract latent representations
def extract_latent_representations(model, data_loader):
    """Extract latent representations from the model for all inputs."""
    model.eval()
    latent_reps = []
    angles_deg = []
    
    with torch.no_grad():
        for inputs, _, angle_vals in data_loader:
            inputs = inputs.to(config.device)
            _, latent = model(inputs)
            latent_reps.append(latent.cpu().numpy())
            angles_deg.append(angle_vals.numpy())
    
    return np.vstack(latent_reps), np.concatenate(angles_deg)

# Function to visualize sample grating stimuli
def visualize_gratings(angles=None):
    """Visualize sample grating stimuli at different orientations with paper-ready formatting."""
    if angles is None:
        # Define angles in degrees as multiples of 22.5°
        angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    
    n = len(angles)
    # Create a 3x3 grid (with the last spot empty)
    rows = 3
    cols = (n + rows - 1) // rows  # Ceiling division to determine columns needed
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.2, rows*1.2), dpi=300)
    axes = axes.flatten()  # Flatten for easier indexing
    
    # Dynamically generate angle labels based on input angles
    angle_labels = [r"${:.0f}^\circ$".format(angle) for angle in angles]
    
    for i, angle in enumerate(angles):
        grating = generate_grating(angle)
        # Use grayscale colormap for better visibility of the gratings
        axes[i].imshow(grating, cmap='gray')
        # Display angle symbolically
        axes[i].set_title(angle_labels[i], fontsize=11)
        axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(angles), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'grating_samples.pdf'), bbox_inches='tight', format='pdf')
    plt.close()

# Function to convert angles to sin/cos representation
def angle_to_sin_cos(angles_deg):
    """Convert angles in degrees to [cos(2θ), sin(2θ)] representation."""
    angles_rad = np.deg2rad(angles_deg)
    cos_vals = np.cos(2 * angles_rad)
    sin_vals = np.sin(2 * angles_rad)
    return np.column_stack((cos_vals, sin_vals))

# Function to convert sin/cos representation back to angles
def sin_cos_to_angle(sin_cos_vals):
    """Convert [cos(2θ), sin(2θ)] representation to angles in degrees."""
    cos_vals = sin_cos_vals[:, 0]
    sin_vals = sin_cos_vals[:, 1]
    angles_rad = 0.5 * np.arctan2(sin_vals, cos_vals)
    angles_deg = np.rad2deg(angles_rad) % 180
    return angles_deg

# Main function
def main():
    print("Generating dataset...")
    neuron_orientations = load_neuron_orientations()
    X, y_angle_deg, orientations = generate_dataset(neuron_orientations)
    # Generate test set with 1 sample from each orientation
    print("Generating test set...")
    
    # Create a test set with one sample from each orientation
    X_test = []
    y_test_angle_deg = []
    orientations_test = []
    
    # Ensure we have one sample for each unique orientation
    unique_orientations = np.unique(orientations)
    
    for angle in unique_orientations:
        # Find indices where this orientation appears
        indices = np.where(orientations == angle)[0]
        
        # Take the first sample with this orientation
        if len(indices) > 0:
            idx = indices[0]
            X_test.append(X[idx])
            y_test_angle_deg.append(y_angle_deg[idx])
            orientations_test.append(orientations[idx])
    
    # Convert to numpy arrays
    X_test = np.array(X_test)
    y_test_angle_deg = np.array(y_test_angle_deg)
    orientations_test = np.array(orientations_test)
    
    print(f"Test set created with {len(X_test)} samples, one from each orientation.")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X)
    y_train_tensor = torch.FloatTensor(y_angle_deg)
    orientations_train_tensor = torch.FloatTensor(orientations)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test_angle_deg)
    orientations_test_tensor = torch.FloatTensor(orientations_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, orientations_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, orientations_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Visualize sample gratings
    print("Visualizing sample gratings...")
    visualize_gratings(neuron_orientations)
    
    # Initialize the model
    print("Initializing model...")
    # Initialize model, optimizer, and loss function
    model = GratingOrientationNet().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()  # MSE works since we normalize output to unit circle

    # Train model
    losses = train_model(model, train_loader, criterion, optimizer)

    # Evaluate model
    angular_error = evaluate_model(model, test_loader)
    print(f"Mean Angular Error: {angular_error:.2f}°")
    
    # Extract latent representations
    print("Extracting latent representations...")
    latent_reps, latent_angles_deg = extract_latent_representations(model, test_loader)
    
    # Apply PCA to get 2D embedding for circular analysis
    pca_embedding = analyze_pca(latent_angles_deg, latent_reps)
    
    # Analyze circular regression
    print("Analyzing circular regression...")
    analyze_circular_regression(latent_angles_deg, pca_embedding, config.results_dir, prefix="ann_")
    
    # Visualize with circular colormap
    print("Visualizing with circular colormap...")
    visualize_circular_colormap(latent_angles_deg, pca_embedding, config.results_dir, prefix="ann_")
    
    print("Simulation complete! Results saved in the 'results' directory.")

if __name__ == "__main__":
    main()
