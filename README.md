# Neural Network Circular Manifold Simulation

This project simulates a neural network learning to predict the orientation of grating stimuli and analyzes the geometry and topology of the network's internal representations to uncover a circular manifold.

## Overview

The simulation demonstrates how a neural network encodes circular data (orientation angles from 0° to 360°) in its latent space. The key insight is that the network should learn a representation that captures the circular topology of the orientation space, where 0° and 360° are the same point.

## Features

- Generates synthetic grating stimuli at various orientations
- Trains a simple convolutional neural network to predict orientation angles
- Analyzes the network's internal representations using dimensionality reduction techniques (PCA, t-SNE, UMAP)
- Visualizes the circular manifold in the latent space
- Fits a circle to the latent representations to quantify the circular structure

## Requirements

See `requirements.txt` for the full list of dependencies. The main requirements are:
- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- UMAP

## Usage

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```
   python circular_manifold_simulation.py
   ```

3. Check the `results` directory for visualizations and analysis outputs.

## Visualizations

The simulation generates several visualizations:
- Sample grating stimuli at different orientations
- Training loss curve
- 2D projections of the latent space using PCA, t-SNE, and UMAP
- 3D visualization of the latent space
- PCA projection with a fitted circle to demonstrate the circular structure

## How It Works

1. **Data Generation**: Creates sinusoidal gratings with orientations uniformly distributed between 0° and 360°.
2. **Angle Representation**: Converts angles to sine and cosine components to handle the circular nature of angles.
3. **Neural Network**: A simple CNN with convolutional layers followed by fully connected layers.
4. **Latent Space Analysis**: Extracts representations from an intermediate layer and applies dimensionality reduction.
5. **Manifold Visualization**: Visualizes the latent space to reveal the circular structure.

## Expected Results

If the network successfully learns the orientation prediction task, the latent representations should form a circular manifold in the high-dimensional space. When projected to 2D or 3D using dimensionality reduction techniques, this structure should be visible as a circle or a circular pattern, with points colored by angle showing a smooth color gradient around the circle. 