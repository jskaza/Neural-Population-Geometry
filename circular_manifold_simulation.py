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
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, MDS
import warnings

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
    max_angle = 2 * np.pi  # in radians (0 to 2π)
    
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
def generate_grating(angle_degrees, size=config.image_size, frequency=5):
    """Generate a sinusoidal grating with the specified orientation angle in degrees.
    
    0 degrees corresponds to a horizontal grating (bars running horizontally).
    90 degrees corresponds to a vertical grating (bars running vertically).
    """
    angle_rad = np.deg2rad(angle_degrees)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Rotate coordinates - adjusted so 0 degrees is horizontal
    # For horizontal grating at 0 degrees, we want to use y directly
    # For vertical grating at 90 degrees, we want to use x directly
    y_rot = xx * np.sin(angle_rad) + yy * np.cos(angle_rad)
    
    # Create grating
    grating = np.sin(2 * np.pi * frequency * y_rot)
    
    # Normalize to [0, 1]
    grating = (grating + 1) / 2
    
    return grating

# Generate dataset
def generate_dataset(num_samples=config.num_samples):
    """Generate a dataset of grating stimuli with random orientations."""
    X = np.zeros((num_samples, 1, config.image_size, config.image_size))
    y_angle_rad = np.zeros(num_samples)
    
    # Generate samples with uniform distribution of angles in radians
    angles_rad = np.random.uniform(config.min_angle, config.max_angle, num_samples)
    
    for i, angle_rad in enumerate(angles_rad):
        # Convert to degrees for the grating generation function
        angle_deg = np.rad2deg(angle_rad)
        X[i, 0] = generate_grating(angle_deg)
        y_angle_rad[i] = angle_rad
    
    # Calculate sine and cosine components directly from radians
    y_sin = np.sin(y_angle_rad)
    y_cos = np.cos(y_angle_rad)
    y = np.column_stack((y_sin, y_cos))
    
    return X, y, y_angle_rad

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
        for inputs, targets, angles_rad in test_loader:
            inputs = inputs.to(config.device)
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Convert predictions (sin, cos) back to angles in radians
            pred_sin = outputs[:, 0].cpu().numpy()
            pred_cos = outputs[:, 1].cpu().numpy()
            pred_angles_rad = np.arctan2(pred_sin, pred_cos) % (2 * np.pi)
            
            # Calculate angular error in radians (considering the circular nature)
            true_angles_rad = angles_rad.numpy()
            errors = np.abs(pred_angles_rad - true_angles_rad)
            errors = np.minimum(errors, 2 * np.pi - errors)
            total_angular_error += np.sum(errors)
    
    # Average error in radians
    avg_angular_error_rad = total_angular_error / len(test_loader.dataset)
    
    # Convert to degrees for reporting (more intuitive for humans)
    avg_angular_error_deg = np.rad2deg(avg_angular_error_rad)
    
    return avg_angular_error_rad, avg_angular_error_deg

# Function to extract latent representations
def extract_latent_representations(model, data_loader):
    """Extract latent representations from the model for all inputs."""
    model.eval()
    latent_reps = []
    angles_rad = []
    
    with torch.no_grad():
        for inputs, _, angle_vals in data_loader:
            inputs = inputs.to(config.device)
            _, latent = model(inputs)
            latent_reps.append(latent.cpu().numpy())
            angles_rad.append(angle_vals.numpy())
    
    return np.vstack(latent_reps), np.concatenate(angles_rad)

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
        # Define angles in radians as multiples of pi/4
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    
    n = len(angles)
    fig, axes = plt.subplots(1, n, figsize=(n*1.2, 1.5), dpi=300)
    
    # Symbolic labels for multiples of pi/4
    angle_labels = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", 
                    r"$\pi$", r"$5\pi/4$", r"$3\pi/2$", r"$7\pi/4$"]
    
    for i, angle in enumerate(angles):
        grating = generate_grating(np.rad2deg(angle))  # Convert to degrees for the existing function
        # Use grayscale colormap for better visibility of the gratings
        axes[i].imshow(grating, cmap='gray')
        # Display angle symbolically
        axes[i].set_title(angle_labels[i], fontsize=11)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'grating_samples.pdf'), bbox_inches='tight', format='pdf')
    plt.close()


# Function to apply and visualize multiple dimensionality reduction techniques
def visualize_multiple_embeddings(latent_reps, angles_rad, filename_prefix='dim_reduction'):
    """
    Apply and visualize multiple dimensionality reduction techniques on the latent representations.
    
    Parameters:
    -----------
    latent_reps : numpy.ndarray
        The high-dimensional latent representations to reduce
    angles_rad : numpy.ndarray
        The corresponding angles for coloring the points
    filename_prefix : str
        Prefix for the output filenames
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
    
    # Sample a subset of points for computational efficiency
    # This is especially important for t-SNE and UMAP which can be slow
    sample_data = latent_reps
    sample_angles_rad = angles_rad
    
    # Define the dimensionality reduction techniques to apply
    techniques = [
        ('PCA', PCA(n_components=2)),
        ('Isomap', Isomap(n_components=2, n_neighbors=15)),
        ('MDS', MDS(n_components=2, random_state=42))
    ]
    
    # Store reduced data for circular structure analysis
    reduced_data_dict = {}
    
    # Define phase angles for markers (from 0 to 2π)
    phase_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    
    # Define more angles for the directional arrows along the trajectory
    arrow_angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    
    # Process each technique and store results
    for name, reducer in techniques:
        print(f"Applying {name}...")
        
        try:
            # Apply dimensionality reduction
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reduced_data = reducer.fit_transform(sample_data)
            
            # Store reduced data for later analysis
            reduced_data_dict[name] = reduced_data
            
            # Save individual plot with directional arrows
            plt.figure(figsize=(5, 4), dpi=300)
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                       c=sample_angles_rad, cmap='Reds', alpha=0.8, s=10)
            
            # Add directional arrows along the circular trajectory
            for i, angle in enumerate(arrow_angles):
                # Find the point closest to this angle
                idx = np.argmin(np.abs(sample_angles_rad - angle))
                point = reduced_data[idx]
                
                # Find the next point (for direction)
                next_angle = arrow_angles[(i + 1) % len(arrow_angles)]
                next_idx = np.argmin(np.abs(sample_angles_rad - next_angle))
                next_point = reduced_data[next_idx]
                
                # Calculate direction vector
                direction = next_point - point
                # Normalize to a consistent length
                arrow_length = 0.1 * np.linalg.norm(direction)
                direction = direction / np.linalg.norm(direction) * arrow_length
                
                # Draw the arrow along the trajectory
                plt.arrow(point[0], point[1], direction[0], direction[1], 
                         head_width=arrow_length*0.4, head_length=arrow_length*0.6, 
                         fc='blue', ec='blue', alpha=0.7)
            
            plt.title(f"{name} of Neural Representations", fontsize=11)
            plt.xlabel(f"{name} 1", fontsize=11)
            plt.ylabel(f"{name} 2", fontsize=11)
            plt.grid(alpha=0.3)
            
            # Create colorbar with π-based ticks
            cbar = plt.colorbar(scatter)
            cbar.set_label(r'Orientation Angle', fontsize=11)
            # Set ticks at multiples of π/4
            tick_locs = np.linspace(0, 2*np.pi, 9)
            tick_labels = [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', 
                          r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$', r'$2\pi$']
            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels(tick_labels)
            
            plt.tight_layout()
            plt.savefig(os.path.join(config.results_dir, f"{filename_prefix}_{name.lower()}.pdf"), 
                       bbox_inches='tight', format='pdf')
            plt.close()
            
        except Exception as e:
            print(f"Error applying {name}: {e}")
    
    # Create combined figure with all techniques in a single row
    n_techniques = len(techniques)
    fig, axes = plt.subplots(1, n_techniques, figsize=(n_techniques*4, 4), dpi=300)
    
    # Apply each technique and visualize in the combined plot
    for i, (name, _) in enumerate(techniques):
        if name not in reduced_data_dict:
            axes[i].text(0.5, 0.5, f"Error applying {name}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            continue
            
        reduced_data = reduced_data_dict[name]
        
        # Create scatter plot with Reds colormap
        scatter = axes[i].scatter(reduced_data[:, 0], reduced_data[:, 1], 
                                 c=sample_angles_rad, cmap='Reds', alpha=0.8, s=10)
        
        # Add directional arrows along the circular trajectory
        for j, angle in enumerate(arrow_angles):
            # Find the point closest to this angle
            idx = np.argmin(np.abs(sample_angles_rad - angle))
            point = reduced_data[idx]
            
            # Find the next point (for direction)
            next_angle = arrow_angles[(j + 1) % len(arrow_angles)]
            next_idx = np.argmin(np.abs(sample_angles_rad - next_angle))
            next_point = reduced_data[next_idx]
            
            # Calculate direction vector
            direction = next_point - point
            # Normalize to a consistent length
            arrow_length = 0.1 * np.linalg.norm(direction)
            direction = direction / np.linalg.norm(direction) * arrow_length
            
            # Draw the arrow along the trajectory
            axes[i].arrow(point[0], point[1], direction[0], direction[1], 
                     head_width=arrow_length*0.4, head_length=arrow_length*0.6, 
                     fc='blue', ec='blue', alpha=0.7)
        
        # Add title and labels
        axes[i].set_title(f"{name}", fontsize=11)
        axes[i].set_xlabel(f"{name} 1", fontsize=11)
        axes[i].set_ylabel(f"{name} 2", fontsize=11)
        
        # Add grid
        axes[i].grid(alpha=0.3)
    
    # Add a single colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label(r'Orientation Angle', fontsize=11)
    
    # Set ticks at multiples of π/4
    tick_locs = np.linspace(0, 2*np.pi, 9)
    tick_labels = [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', 
                  r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$', r'$2\pi$']
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_labels)
    
    # Improve layout and save the combined figure
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
    plt.savefig(os.path.join(config.results_dir, f"{filename_prefix}_combined.pdf"), 
               bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"Dimensionality reduction visualizations saved to {config.results_dir}")
    
    return reduced_data_dict

# Main function
def main():
    print("Generating dataset...")
    X, y, angles_rad = generate_dataset()
    
    # Split into train and test sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    angles_train_rad, angles_test_rad = angles_rad[:split_idx], angles_rad[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    angles_train_tensor = torch.FloatTensor(angles_train_rad)
    angles_test_tensor = torch.FloatTensor(angles_test_rad)
    
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
    train_model(model, train_loader, criterion, optimizer)
    
    
    # Evaluate the model
    print("Evaluating model...")
    avg_angular_error_rad, avg_angular_error_deg = evaluate_model(model, test_loader)
    print(f"Average Angular Error (radians): {avg_angular_error_rad:.4f}")
    print(f"Average Angular Error (degrees): {avg_angular_error_deg:.2f}")
    
    # Extract latent representations
    print("Extracting latent representations...")
    latent_reps, latent_angles_rad = extract_latent_representations(model, test_loader)
    
    
    # Apply and visualize multiple dimensionality reduction techniques
    print("Applying multiple dimensionality reduction techniques...")
    reduced_data_dict = visualize_multiple_embeddings(latent_reps, latent_angles_rad)

    print("Simulation complete! Results saved in the 'results' directory.")

if __name__ == "__main__":
    main()
