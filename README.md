# Neural Network and Biological Neural Circuit Circular Manifold Simulation

This project simulates a neural network learning to predict the orientation of grating stimuli and analyzes the geometry and topology of the network's internal representations to uncover a circular manifold. Additionally, it examines similar structures in biological neural circuits using mouse visual cortex data.

## Overview

The project demonstrates how both artificial and biological neural networks encode circular data (orientation angles from 0° to 360°) in their latent spaces. The key insight is that these networks should learn representations that capture the circular topology of the orientation space.

## Features

- Generates synthetic grating stimuli at various orientations
- Trains a simple convolutional neural network to predict orientation angles
- Analyzes the network's internal representations using dimensionality reduction techniques (PCA, t-SNE, UMAP)
- Visualizes the circular manifold in the latent space
- Fits a circle to the latent representations to quantify the circular structure
- Analyzes mouse visual cortex data to compare biological and artificial neural representations

## Requirements

The project requires Python 3.6+ and the following packages:

- NumPy
- SciPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch
- tqdm
- UMAP-learn
- Ripser
- Persim
- Gudhi

For LaTeX rendering in plots, ensure LaTeX is installed on your system.

## Usage

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```bash
   python ann_manifold_simulation.py
   python neuron_manifold_analysis.py
   ```

3. Check the `results` directory for visualizations and analysis outputs.

## Visualizations

The simulation generates several visualizations:
- Sample grating stimuli at different orientations
- 2D projections of the latent space
- PCA projection with a fitted circle to demonstrate the circular structure
- Analysis of mouse visual cortex data showing similar circular manifolds