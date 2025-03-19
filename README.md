# Neural Network and Biological Neural Circuit Circular Manifold Simulation


This is a class project for *ME 225NN: Modeling and Optimization of Neural Networks* at UC Santa Barbara and is based on [Chung & Abbott (2021)](https://www.sciencedirect.com/science/article/pii/S0959438821001227#bib32). We simulate a neural network learning to predict the orientation of grating stimuli and analyze the geometry and topology of the network's internal representations to uncover a circular manifold. Additionally, it examines similar structures in biological neural circuits using mouse visual cortex recordings.

## Overview

The project demonstrates how both artificial and biological neural networks encode circular data (orientation angles from 0° to 360°) in their latent spaces. See `main.pdf` for the write-up.

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


