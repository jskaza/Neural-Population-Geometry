#!/usr/bin/env python3
"""
Manifold Analysis Utilities

This module provides standardized functions for analyzing neural manifolds,
particularly for circular manifold analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import os

# Set up LaTeX rendering for plots with size-independent fonts
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,          # Base font size
    'axes.labelsize': 10,     # Base label size
    'axes.titlesize': 11,     # Slightly larger for titles
    'xtick.labelsize': 9,     # Slightly smaller for ticks
    'ytick.labelsize': 9,     # Slightly smaller for ticks
    'legend.fontsize': 9,     # Slightly smaller for legend
    'text.usetex': True,      # Enable LaTeX rendering
    'text.latex.preamble': r'\usepackage{amsmath}',
    'figure.dpi': 300,        # High DPI for the figure object
    'savefig.dpi': 300,       # High DPI for saved figures
    'figure.autolayout': True # Better layout handling
})

def analyze_pca(orientations, population_responses):
    """
    Apply PCA to check for ring-like structure without assuming periodicity.
    
    Parameters:
    -----------
    orientations : numpy.ndarray
        Array of orientation angles in degrees
    population_responses : numpy.ndarray
        Matrix of population responses, shape (n_orientations, n_neurons)
        
    Returns:
    --------
    reduced_data : numpy.ndarray
        PCA-reduced data, shape (n_orientations, n_components)
    """    
    # Apply PCA
    pca = PCA(n_components=5)  # Get more components to check variance distribution
    reduced_data = pca.fit_transform(population_responses)
    explained_variance = pca.explained_variance_ratio_
    
    # Print variance explained
    print("\nVariance explained by principal components:")
    for i, var in enumerate(explained_variance[:5]):
        print(f"PC{i+1}: {var:.3f}")
    
    return reduced_data

def analyze_circular_regression(orientations, embedding, results_dir, prefix=""):
    """
    Fit a model of the form: y = a*cos(2θ) + b*sin(2θ) + c
    to test for circular structure in the embedding.
    
    Parameters:
    -----------
    orientations : numpy.ndarray
        Array of orientation angles in degrees
    embedding : numpy.ndarray
        Embedding data, typically from PCA, shape (n_orientations, n_components)
    results_dir : str
        Directory to save results
    prefix : str, optional
        Prefix for output files, e.g., "mouse_" or "ann_"
        
    Returns:
    --------
    results : dict
        Dictionary containing regression results
    """    
    # Define circular fit function
    def circular_fit(theta, a, b, c):
        return a * np.cos(2 * theta) + b * np.sin(2 * theta) + c
    
    # Convert orientations to radians
    theta_radians = np.deg2rad(orientations)
    
    # Create figure with size optimized for 0.48 textwidth
    fig = plt.figure(figsize=(4, 4))  # Width matches 0.48 textwidth, height adjusted for 2 subplots
    
    # Create subplots for the two components
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    
    # Sort data by orientation for cleaner visualization
    sort_idx = np.argsort(orientations)
    sorted_orientations = orientations[sort_idx]
    sorted_data = embedding[sort_idx]
    sorted_theta = theta_radians[sort_idx]
    
    results = {}
    
    try:
        # Fit circular model to first component
        params1, _ = curve_fit(circular_fit, theta_radians, embedding[:, 0])
        fitted_values1 = circular_fit(sorted_theta, *params1)
        
        # Calculate R² for the fit
        ss_total1 = np.sum((embedding[:, 0] - np.mean(embedding[:, 0]))**2)
        ss_residual1 = np.sum((embedding[:, 0] - circular_fit(theta_radians, *params1))**2)
        r_squared1 = 1 - (ss_residual1 / ss_total1)
        
        # Plot original data and fit for component 1
        ax1.scatter(sorted_orientations, sorted_data[:, 0], 
                    c=sorted_orientations, cmap='Reds', label='Data')
        ax1.plot(sorted_orientations, fitted_values1, 'k--', linewidth=2, 
                label=r'Circular Fit ($R^2={:.3f}$)'.format(r_squared1))
        
        ax1.set_title(r'PC 1 Circular Fit')
        ax1.set_xlabel(r'Orientation ($^\circ$)')
        ax1.set_ylabel(r'PC 1')
        ax1.legend()
        
        # Fit circular model to second component
        params2, _ = curve_fit(circular_fit, theta_radians, embedding[:, 1])
        fitted_values2 = circular_fit(sorted_theta, *params2)
        
        # Calculate R² for the fit
        ss_total2 = np.sum((embedding[:, 1] - np.mean(embedding[:, 1]))**2)
        ss_residual2 = np.sum((embedding[:, 1] - circular_fit(theta_radians, *params2))**2)
        r_squared2 = 1 - (ss_residual2 / ss_total2)
        
        # Plot original data and fit for component 2
        ax2.scatter(sorted_orientations, sorted_data[:, 1], 
                    c=sorted_orientations, cmap='Reds', label='Data')
        ax2.plot(sorted_orientations, fitted_values2, 'k--', linewidth=2, 
                label=r'Circular Fit ($R^2={:.3f}$)'.format(r_squared2))
        
        ax2.set_title(r'PC 2 Circular Fit')
        ax2.set_xlabel(r'Orientation ($^\circ$)')
        ax2.set_ylabel(r'PC 2')
        ax2.legend()

        # Combined R²
        combined_r_squared = np.sqrt(r_squared1**2 + r_squared2**2) / np.sqrt(2)
        
        # Store results
        results = {
            'params1': params1,
            'params2': params2,
            'r_squared1': r_squared1,
            'r_squared2': r_squared2,
            'combined_r_squared': combined_r_squared,
        }
        
    except RuntimeError:
        ax1.text(0.5, 0.5, "Fitting failed", 
                ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "Fitting failed", 
                ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{prefix}circular_regression.pdf"))
    plt.close()
    
    # Print results
    print("\nCircular regression results:")
    if 'params1' in results:
        print(f"  Component 1: R²={results['r_squared1']:.3f}, params={results['params1']}")
        print(f"  Component 2: R²={results['r_squared2']:.3f}, params={results['params2']}")
        print(f"  Combined R²: {results['combined_r_squared']:.3f}")
    
    return results

def visualize_circular_colormap(orientations, embedding, results_dir, prefix=""):
    """
    Visualize PCA embeddings with a circular colormap to confirm if responses wrap around smoothly.
    Uses Reds colormap where hue directly corresponds to orientation angle.
    
    Parameters:
    -----------
    orientations : numpy.ndarray
        Array of orientation angles in degrees
    embedding : numpy.ndarray
        Embedding data, typically from PCA, shape (n_orientations, n_components)
    results_dir : str
        Directory to save results
    prefix : str, optional
        Prefix for output files, e.g., "mouse_" or "ann_"
    """
    print("\nVisualizing embeddings with circular colormap...")
    
    # Create figure with size optimized for 0.48 textwidth in LaTeX
    # For a typical LaTeX document, textwidth is about 6.5 inches
    # So 0.48 textwidth is about 3.12 inches
    fig = plt.figure(figsize=(4, 4))  # Square figure that's 0.48 textwidth
    
    norm_orientations = orientations / 360.0
    
    # Sort by orientation for visualization
    sort_idx = np.argsort(orientations)
    sorted_data = embedding[sort_idx, :2]  # Use first two components
    sorted_orientations = orientations[sort_idx]
    
    # Create custom colormap that wraps around
    colors = plt.cm.Reds(norm_orientations)
    
    # Plot points with orientation-based color
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=100, alpha=0.8)
    
    # Connect points in sequence to show trajectory
    plt.plot(sorted_data[:, 0], sorted_data[:, 1], 'k-', alpha=0.5, linewidth=1)
    
    # Add directional arrows along the trajectory
    for i in range(len(sorted_data) - 1):
        # Calculate direction vector
        start_point = sorted_data[i]
        end_point = sorted_data[i + 1]
        direction = end_point - start_point
        
        # Use fixed arrow length instead of normalizing
        arrow_length = 0.3  # Fixed length for all arrows
        
        # Calculate unit direction vector
        unit_direction = direction / np.linalg.norm(direction)
        
        # Create arrow with fixed length
        arrow_vector = unit_direction * arrow_length
        
        # Draw the arrow
        plt.arrow(start_point[0], start_point[1], arrow_vector[0], arrow_vector[1],
                  head_width=arrow_length*0.4, head_length=arrow_length*0.6,
                  fc='blue', ec='blue', alpha=0.7)
    
    # Add orientation labels with jittering to prevent overlap
    # Calculate the average distance between points to determine jitter scale
    distances = []
    for i in range(len(sorted_data) - 1):
        distances.append(np.linalg.norm(sorted_data[i+1] - sorted_data[i]))
    jitter_scale = np.mean(distances) * 0.2  # Use 20% of average distance for jitter
    
    # Use angle-based jittering to spread labels out
    for i, angle in enumerate(sorted_orientations):
        # Convert angle to radians for positioning calculation
        theta = np.deg2rad(angle)
        # Apply small offset based on angle
        x_jitter = jitter_scale * np.cos(theta)
        y_jitter = jitter_scale * np.sin(theta)
        
        # Position text with jitter and slight outward offset
        plt.text(sorted_data[i, 0] + x_jitter, sorted_data[i, 1] + y_jitter, 
               f"${int(angle)}^\\circ$", 
               fontsize=8, ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Add a colorbar with orientation ticks below the plot
    # Adjust the figure to make room for the colorbar
    plt.subplots_adjust(bottom=0.15)  # Increase bottom margin
    
    # # Position the colorbar below the plot with more space
    # cax = plt.axes([0.15, 0.07, 0.7, 0.03])  # [left, bottom, width, height] - increased bottom value
    # cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='Reds'), cax=cax, orientation='horizontal')
    # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    # cbar.set_ticklabels(['$0^\\circ$', '$90^\\circ$', '$180^\\circ$', '$270^\\circ$', '$360^\\circ$'])
    # cbar.set_label('Orientation')
    plt.title(r'PCA Embedding')
    plt.xlabel(r'PC 1')
    plt.ylabel(r'PC 2')
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{prefix}circular_colormap_visualization.pdf"))
    plt.close() 