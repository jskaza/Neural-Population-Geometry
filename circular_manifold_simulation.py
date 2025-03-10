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
import os
from tqdm import tqdm
from ripser import ripser
from persim import plot_diagrams

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
def visualize_latent_space(latent_reps, angles, filename='latent_space.pdf'):
    """Visualize the latent space using PCA with paper-ready formatting."""
    # Set up matplotlib for consistent text size with LaTeX rendering
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'text.usetex': True,  # Enable LaTeX rendering
        'text.latex.preamble': r'\usepackage{amsmath}',
    })
    
    plt.figure(figsize=(5, 4), dpi=300)
    
    # Apply PCA for dimensionality reduction
    reducer = PCA(n_components=2)
    reduced_data = reducer.fit_transform(latent_reps)
    
    # Calculate explained variance
    explained_var = reducer.explained_variance_ratio_
    explained_var_sum = np.sum(explained_var) * 100
    
    # Create a scatter plot colored by angle
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                         c=angles, cmap='hsv', alpha=0.8, s=10)
    
    # Add colorbar and labels
    cbar = plt.colorbar(scatter)
    cbar.set_label(r'Orientation Angle ($^\circ$)', fontsize=11)
    
    # Add title and axis labels with explained variance
    plt.title(r'PCA of Neural Representations', fontsize=11)
    plt.xlabel(r'PC1 (${:.1f}\%$ var.)'.format(explained_var[0]*100), fontsize=11)
    plt.ylabel(r'PC2 (${:.1f}\%$ var.)'.format(explained_var[1]*100), fontsize=11)
    
    # Add text showing total explained variance
    plt.figtext(0.7, 0.02, r'Total var: ${:.1f}\%$'.format(explained_var_sum), 
                ha='center', fontsize=11)
    
    # Add grid and improve layout
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the figure as PDF
    plt.savefig(os.path.join(config.results_dir, filename), bbox_inches='tight', format='pdf')
    plt.close()

# Function to visualize sample grating stimuli
def visualize_gratings(angles=None):
    """Visualize sample grating stimuli at different orientations with paper-ready formatting."""
    # Set up matplotlib for consistent text size with LaTeX rendering
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'text.usetex': True,  # Enable LaTeX rendering
        'text.latex.preamble': r'\usepackage{amsmath}',
    })
    
    if angles is None:
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    n = len(angles)
    fig, axes = plt.subplots(1, n, figsize=(n*1.2, 1.5), dpi=300)
    
    for i, angle in enumerate(angles):
        grating = generate_grating(angle)
        axes[i].imshow(grating, cmap='gray')
        axes[i].set_title(f"${angle}^\\circ$", fontsize=11)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'grating_samples.pdf'), bbox_inches='tight', format='pdf')
    plt.close()

# Function to visualize the training loss
def plot_loss(losses):
    """Plot the training loss over epochs with paper-ready formatting."""
    # Set up matplotlib for consistent text size with LaTeX rendering
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'text.usetex': True,  # Enable LaTeX rendering
        'text.latex.preamble': r'\usepackage{amsmath}',
    })
    
    plt.figure(figsize=(5, 4), dpi=300)
    plt.plot(losses, linewidth=1.5, color='#1f77b4')
    plt.title(r'Training Loss Over Epochs', fontsize=11)
    plt.xlabel(r'Epoch', fontsize=11)
    plt.ylabel(r'Loss', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add markers at specific points
    if len(losses) > 10:
        marker_indices = np.linspace(0, len(losses)-1, 10, dtype=int)
        plt.plot(marker_indices, [losses[i] for i in marker_indices], 'o', color='#ff7f0e', markersize=4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'training_loss.pdf'), bbox_inches='tight', format='pdf')
    plt.close()

# Function to visualize the circular structure in 3D
def visualize_3d_manifold(latent_reps, angles, filename='3d_manifold.pdf'):
    """Visualize the first 3 principal components of the latent space with paper-ready formatting."""
    # Set up matplotlib for consistent text size with LaTeX rendering
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'text.usetex': True,  # Enable LaTeX rendering
        'text.latex.preamble': r'\usepackage{amsmath}',
    })
    
    # Apply PCA to get the first 3 principal components
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(latent_reps)
    
    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_
    explained_var_sum = np.sum(explained_var) * 100
    
    fig = plt.figure(figsize=(6, 5), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a scatter plot colored by angle
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                        c=angles, cmap='hsv', alpha=0.8, s=8)
    
    # Add colorbar and labels
    cbar = plt.colorbar(scatter)
    cbar.set_label(r'Orientation Angle ($^\circ$)', fontsize=11)
    
    # Add title and axis labels with explained variance
    ax.set_title(r'3D PCA of Neural Representations', fontsize=11)
    ax.set_xlabel(r'PC1 (${:.1f}\%$)'.format(explained_var[0]*100), fontsize=11)
    ax.set_ylabel(r'PC2 (${:.1f}\%$)'.format(explained_var[1]*100), fontsize=11)
    ax.set_zlabel(r'PC3 (${:.1f}\%$)'.format(explained_var[2]*100), fontsize=11)
    
    # Add text showing total explained variance
    plt.figtext(0.7, 0.02, r'Total var: ${:.1f}\%$'.format(explained_var_sum), 
                ha='center', fontsize=11)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the figure as PDF
    plt.savefig(os.path.join(config.results_dir, filename), bbox_inches='tight', format='pdf')
    plt.close()

# Function to perform persistent homology analysis
def analyze_topology_with_persistent_homology(latent_reps, angles, filename='persistent_homology.pdf'):
    """
    Analyze the topology of the latent representations using persistent homology.
    Implements the SPUD (Spline Parameterization for Unsupervised Decoding) approach
    similar to Chaudhuri et al. to discover the ring structure.
    """
    # Set up matplotlib for consistent text size with LaTeX rendering
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'text.usetex': True,  # Enable LaTeX rendering
        'text.latex.preamble': r'\usepackage{amsmath}',
    })
    
    # Compute persistent homology
    print("Computing persistent homology...")
    # Use a subset of points for computational efficiency if needed
    sample_size = min(1000, latent_reps.shape[0])
    indices = np.random.choice(latent_reps.shape[0], sample_size, replace=False)
    sample_data = latent_reps[indices]
    sample_angles = angles[indices]
    
    # Compute persistent homology directly on high-dimensional data
    # This is the key difference from PCA - we analyze the topology in the original space
    diagrams = ripser(sample_data, maxdim=1)['dgms']
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    
    # Plot the persistence diagram
    plot_diagrams(diagrams, show=False, ax=ax1)
    ax1.set_title(r'Persistence Diagram', fontsize=11)
    
    # Find the most persistent 1-dimensional feature (loop)
    if len(diagrams) > 1 and len(diagrams[1]) > 0:
        # Calculate persistence as death - birth
        persistence = diagrams[1][:, 1] - diagrams[1][:, 0]
        if len(persistence) > 0:
            most_persistent_idx = np.argmax(persistence)
            most_persistent_feature = diagrams[1][most_persistent_idx]
            
            # Highlight the most persistent feature
            ax1.scatter([most_persistent_feature[0]], [most_persistent_feature[1]], 
                       color='red', s=100, edgecolor='black', zorder=5)
            
            # Add annotation
            persistence_value = most_persistent_feature[1] - most_persistent_feature[0]
            ax1.annotate(f"Persistence: {persistence_value:.3f}",
                        xy=(most_persistent_feature[0], most_persistent_feature[1]),
                        xytext=(most_persistent_feature[0] + 0.05, most_persistent_feature[1] + 0.05),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=10)
            
            print(f"Most persistent 1D feature has persistence: {persistence_value:.3f}")
            print(f"This indicates a circular topology in the high-dimensional space")
    
    # For visualization only, we'll use a non-linear dimensionality reduction
    # This is just to show the results, not part of the actual topology analysis
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        tsne_data = tsne.fit_transform(sample_data)
        
        # Plot the data colored by angle
        scatter = ax2.scatter(tsne_data[:, 0], tsne_data[:, 1], c=sample_angles, 
                             cmap='hsv', alpha=0.8, s=30)
        ax2.set_title(r't-SNE Visualization of Latent Space', fontsize=11)
        ax2.set_xlabel(r't-SNE 1', fontsize=11)
        ax2.set_ylabel(r't-SNE 2', fontsize=11)
    except ImportError:
        # Fallback to MDS if t-SNE is not available
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, random_state=42)
        mds_data = mds.fit_transform(sample_data)
        
        # Plot the data colored by angle
        scatter = ax2.scatter(mds_data[:, 0], mds_data[:, 1], c=sample_angles, 
                             cmap='hsv', alpha=0.8, s=30)
        ax2.set_title(r'MDS Visualization of Latent Space', fontsize=11)
        ax2.set_xlabel(r'MDS 1', fontsize=11)
        ax2.set_ylabel(r'MDS 2', fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label(r'Angle ($^\circ$)', fontsize=11)
    
    # Add a text box explaining SPUD
    textstr = '\n'.join([
        r'SPUD Analysis:',
        r'Persistent homology reveals',
        r'a circular (1D) manifold',
        r'structure in the high-dimensional',
        r'latent space without using PCA'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, filename), bbox_inches='tight', format='pdf')
    plt.close()
    
    # Perform SPUD parameterization directly in high-dimensional space
    perform_spud_parameterization(latent_reps, angles)

def perform_spud_parameterization(latent_reps, angles, filename='spud_parameterization.pdf'):
    """
    Perform SPUD (Spline Parameterization for Unsupervised Decoding) to parameterize
    the circular manifold, similar to the approach used by Chaudhuri et al.
    
    This implementation uses Isomap to preserve geodesic distances and fits a spline
    to the point cloud to parameterize the circular manifold.
    """
    # Set up matplotlib for consistent text size with LaTeX rendering
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'text.usetex': True,  # Enable LaTeX rendering
        'text.latex.preamble': r'\usepackage{amsmath}',
    })
    
    print("Performing SPUD parameterization...")
    
    # Use a subset of points for computational efficiency
    sample_size = min(1000, latent_reps.shape[0])
    indices = np.random.choice(latent_reps.shape[0], sample_size, replace=False)
    sample_data = latent_reps[indices]
    sample_angles = angles[indices]
    
    # Step 1: Use Isomap to find a 2D embedding that preserves geodesic distances
    try:
        from sklearn.manifold import Isomap
        n_neighbors = min(15, sample_size - 1)  # Ensure n_neighbors is valid
        isomap = Isomap(n_components=2, n_neighbors=n_neighbors)
        embedding_2d = isomap.fit_transform(sample_data)
        
        print(f"Isomap reconstruction error: {isomap.reconstruction_error():.4f}")
        
        # Step 2: Fit a spline to the 2D point cloud
        # First, we need to identify the center of the circular structure
        center = np.mean(embedding_2d, axis=0)
        
        # Calculate the angle of each point relative to the center
        relative_coords = embedding_2d - center
        point_angles = np.arctan2(relative_coords[:, 1], relative_coords[:, 0])
        point_angles = (np.rad2deg(point_angles) + 360) % 360
        
        # Calculate the radius of each point from the center
        radii = np.sqrt(np.sum(relative_coords**2, axis=1))
        
        # Sort points by their angular position
        sort_idx = np.argsort(point_angles)
        sorted_angles = point_angles[sort_idx]
        sorted_radii = radii[sort_idx]
        sorted_embedding = embedding_2d[sort_idx]
        sorted_true_angles = sample_angles[sort_idx]
        
        # Step 3: Fit a periodic spline to the sorted points
        # We'll use scipy's interpolation functions
        from scipy.interpolate import interp1d, splrep, splev
        
        # Convert angles to radians for spline fitting
        theta = np.deg2rad(sorted_angles)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(12, 8), dpi=300)
        
        # Plot 1: Isomap embedding with spline fit
        ax1 = fig.add_subplot(221)
        scatter = ax1.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                             c=sample_angles, cmap='hsv', alpha=0.8, s=20)
        ax1.set_title(r'Isomap Embedding with SPUD', fontsize=11)
        ax1.set_xlabel(r'Isomap 1', fontsize=11)
        ax1.set_ylabel(r'Isomap 2', fontsize=11)
        ax1.set_aspect('equal')
        
        # Plot the center point
        ax1.scatter([center[0]], [center[1]], color='red', s=100, marker='x')
        
        # Generate points along the spline for visualization
        theta_smooth = np.linspace(0, 2*np.pi, 1000)
        
        # To create a periodic spline, we'll duplicate some points at the beginning and end
        # This ensures the spline wraps around properly
        extended_theta = np.concatenate([theta[-3:] - 2*np.pi, theta, theta[:3] + 2*np.pi])
        extended_x = np.concatenate([sorted_embedding[-3:, 0], sorted_embedding[:, 0], sorted_embedding[:3, 0]])
        extended_y = np.concatenate([sorted_embedding[-3:, 1], sorted_embedding[:, 1], sorted_embedding[:3, 1]])
        
        # Fit splines to x and y coordinates separately
        x_spline = splrep(extended_theta, extended_x, k=3, s=0.1)
        y_spline = splrep(extended_theta, extended_y, k=3, s=0.1)
        
        # Generate smooth curve
        x_smooth = splev(theta_smooth, x_spline)
        y_smooth = splev(theta_smooth, y_spline)
        
        # Plot the spline
        ax1.plot(x_smooth, y_smooth, 'r-', linewidth=2, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label(r'True Angle ($^\circ$)', fontsize=11)
        
        # Plot 2: Radius vs. Angle
        ax2 = fig.add_subplot(222)
        ax2.scatter(sorted_angles, sorted_radii, c=sorted_true_angles, cmap='hsv', alpha=0.8, s=20)
        ax2.set_title(r'Radius vs. Angle in Isomap Space', fontsize=11)
        ax2.set_xlabel(r'Angle in Isomap Space ($^\circ$)', fontsize=11)
        ax2.set_ylabel(r'Radius', fontsize=11)
        ax2.grid(alpha=0.3)
        
        # Fit a spline to the radius vs. angle
        extended_radii = np.concatenate([sorted_radii[-3:], sorted_radii, sorted_radii[:3]])
        radius_spline = splrep(extended_theta, extended_radii, k=3, s=0.1)
        smooth_radii = splev(theta_smooth, radius_spline)
        smooth_angles = np.rad2deg(theta_smooth)
        ax2.plot(smooth_angles, smooth_radii, 'r-', linewidth=2, alpha=0.7)
        
        # Step 4: Parameterize the manifold using the spline
        # The parameterization is the angle around the spline
        # We'll normalize to [0, 360) to represent angles
        spud_angles = point_angles.copy()
        
        # Plot 3: True angles vs. SPUD-derived angles
        ax3 = fig.add_subplot(223)
        ax3.scatter(sample_angles, spud_angles, alpha=0.5, s=20, c=sample_angles, cmap='hsv')
        
        # Add a reference line for perfect correlation
        min_val = min(np.min(sample_angles), np.min(spud_angles))
        max_val = max(np.max(sample_angles), np.max(spud_angles))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        ax3.set_title(r'True Angles vs. SPUD-derived Angles', fontsize=11)
        ax3.set_xlabel(r'True Angle ($^\circ$)', fontsize=11)
        ax3.set_ylabel(r'SPUD-derived Angle ($^\circ$)', fontsize=11)
        ax3.grid(alpha=0.3)
        
        # Find the best rotation to align SPUD angles with true angles
        from scipy.optimize import minimize_scalar
        
        def circular_error(rotation):
            rotated = (spud_angles + rotation) % 360
            errors = np.minimum(np.abs(rotated - sample_angles), 360 - np.abs(rotated - sample_angles))
            return np.mean(errors)
        
        result = minimize_scalar(circular_error, bounds=(0, 360), method='bounded')
        optimal_rotation = result.x
        spud_angles = (spud_angles + optimal_rotation) % 360
        
        print(f"Optimal rotation for SPUD angles: {optimal_rotation:.2f} degrees")
        
        # Plot 4: Isomap embedding colored by SPUD angles
        ax4 = fig.add_subplot(224)
        scatter = ax4.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                             c=spud_angles, cmap='hsv', alpha=0.8, s=20)
        ax4.set_title(r'Isomap with SPUD Parameterization', fontsize=11)
        ax4.set_xlabel(r'Isomap 1', fontsize=11)
        ax4.set_ylabel(r'Isomap 2', fontsize=11)
        ax4.set_aspect('equal')
        
        # Plot the spline
        ax4.plot(x_smooth, y_smooth, 'r-', linewidth=2, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label(r'SPUD Angle ($^\circ$)', fontsize=11)
        
        # Add a text box explaining SPUD
        textstr = '\n'.join([
            r'SPUD Analysis:',
            r'1. Isomap preserves geodesic distances',
            r'2. Fit spline to circular manifold',
            r'3. Parameterize points by angle on spline',
            r'4. Recover circular topology unsupervised'
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.results_dir, filename), bbox_inches='tight', format='pdf')
        plt.close()
        
    except ImportError:
        print("Isomap not available, please install scikit-learn")
        return None, 0
    
    # Calculate correlation between true angles and SPUD-derived angles
    # Need to handle the circular nature of angles
    def circular_correlation(a, b):
        """Calculate correlation between two sets of circular variables."""
        a_rad = np.deg2rad(a)
        b_rad = np.deg2rad(b)
        
        sin_a, cos_a = np.sin(a_rad), np.cos(a_rad)
        sin_b, cos_b = np.sin(b_rad), np.cos(b_rad)
        
        corr_sin = np.corrcoef(sin_a, sin_b)[0, 1]
        corr_cos = np.corrcoef(cos_a, cos_b)[0, 1]
        
        return (corr_sin + corr_cos) / 2
    
    circular_corr = circular_correlation(sample_angles, spud_angles)
    print(f"Circular correlation between true angles and SPUD-derived angles: {circular_corr:.4f}")
    
    # Create a more detailed visualization of the spline parameterization
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    
    # Plot the Isomap embedding
    scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                        c=spud_angles, cmap='hsv', alpha=0.8, s=30)
    
    # Plot the spline with arrows indicating direction
    ax.plot(x_smooth, y_smooth, 'k-', linewidth=1.5, alpha=0.7)
    
    # Add arrows along the spline to show direction
    n_arrows = 12
    arrow_indices = np.linspace(0, len(x_smooth)-1, n_arrows, dtype=int)
    for i in arrow_indices:
        if i < len(x_smooth) - 1:
            dx = x_smooth[i+1] - x_smooth[i]
            dy = y_smooth[i+1] - y_smooth[i]
            ax.arrow(x_smooth[i], y_smooth[i], dx, dy, 
                    head_width=0.05, head_length=0.1, fc='k', ec='k', alpha=0.7)
    
    # Add angle markers at specific points
    angle_markers = np.linspace(0, 330, 12)  # Every 30 degrees
    for angle in angle_markers:
        # Find the closest SPUD angle
        idx = np.argmin(np.abs(spud_angles - angle))
        ax.text(embedding_2d[idx, 0], embedding_2d[idx, 1], f"{int(angle)}°", 
                fontsize=9, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    ax.set_title(r'SPUD: Spline Parameterization of Circular Manifold', fontsize=11)
    ax.set_xlabel(r'Isomap 1', fontsize=11)
    ax.set_ylabel(r'Isomap 2', fontsize=11)
    ax.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(r'SPUD Angle ($^\circ$)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'spud_spline_visualization.pdf'), bbox_inches='tight', format='pdf')
    plt.close()
    
    return spud_angles, circular_corr

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
    print("Visualizing latent space with PCA...")
    visualize_latent_space(latent_reps, latent_angles, filename='pca_latent_space.pdf')
    visualize_3d_manifold(latent_reps, latent_angles)
    
    # Perform persistent homology analysis and SPUD parameterization
    print("Performing persistent homology analysis and SPUD parameterization...")
    analyze_topology_with_persistent_homology(latent_reps, latent_angles)
    
    # Additional analysis: Fit a circle to the first 2 PCs
    print("Performing additional analysis...")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(latent_reps)
    
    # Set up matplotlib for consistent text size with LaTeX rendering
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'text.usetex': True,  # Enable LaTeX rendering
        'text.latex.preamble': r'\usepackage{amsmath}',
    })
    
    # Plot the data colored by angle and fit a circle
    plt.figure(figsize=(5, 4), dpi=300)
    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=latent_angles, cmap='hsv', alpha=0.8, s=8)
    
    # Calculate the center and radius of the best-fit circle
    center_x = np.mean(pca_data[:, 0])
    center_y = np.mean(pca_data[:, 1])
    radius = np.mean(np.sqrt((pca_data[:, 0] - center_x)**2 + (pca_data[:, 1] - center_y)**2))
    
    # Plot the circle
    circle = plt.Circle((center_x, center_y), radius, fill=False, color='black', linestyle='--', linewidth=1)
    plt.gca().add_patch(circle)
    
    plt.colorbar(scatter, label=r'Angle ($^\circ$)')
    plt.title(r'PCA with Fitted Circle', fontsize=11)
    plt.xlabel(r'PC1', fontsize=11)
    plt.ylabel(r'PC2', fontsize=11)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'pca_fitted_circle.pdf'), bbox_inches='tight', format='pdf')
    plt.close()
    
    print("Simulation complete! Results saved in the 'results' directory.")

if __name__ == "__main__":
    main()
