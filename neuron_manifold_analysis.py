#!/usr/bin/env python3
"""
Neuron Manifold Analysis - Circularity Analysis
Tests for emergent circularity in neural population responses to oriented gratings
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from manifold_utils import analyze_pca, analyze_circular_regression, visualize_circular_colormap

# Configuration
DATA_FILE = 'cell_properties.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    """Load and preprocess neural data."""
    print("Loading and preprocessing data...")
    
    # Load the data
    df = pd.read_csv(DATA_FILE)
    
    # Parse string arrays
    df['orientations'] = df['orientations'].apply(lambda s: [float(x) for x in s.strip('[]').split()])
    df['mean_responses'] = df['mean_responses'].apply(lambda s: [float(x) for x in s.strip('[]').split()])
    
    # Filter cells based on quality metrics
    filtered_df = df[(df['osi'] >= 0.5) & (df['r_squared'] >= 0.6)]
    print(f"Filtered from {len(df)} to {len(filtered_df)} cells based on criteria.")
    
    # Extract data
    orientations = np.array(filtered_df['orientations'].iloc[0])
    responses = np.zeros((len(filtered_df), len(orientations)))
    for i, resp in enumerate(filtered_df['mean_responses']):
        responses[i] = np.array(resp)
    
    return orientations, responses

def normalize_responses(responses):
    """Normalize responses to [0,1] range for each neuron."""
    return (responses - responses.min(axis=1)[:, None]) / (responses.max(axis=1) - responses.min(axis=1))[:, None]

def main():
    # Load and process data
    orientations, responses = load_data()
    
    # Normalize responses and create population response matrix
    normalized_responses = normalize_responses(responses)
    population_responses = normalized_responses.T
    
    # Step 1: Baseline PCA
    embedding = analyze_pca(orientations, population_responses)
    
    # Step 2: Circular regression test
    analyze_circular_regression(orientations, embedding, RESULTS_DIR, prefix="mouse_")
    
    # Step 3: Visualize with circular colormap
    visualize_circular_colormap(orientations, embedding, RESULTS_DIR, prefix="mouse_")
    
    print("\nAnalysis complete! Results saved in the 'results' directory.")

if __name__ == "__main__":
    main() 