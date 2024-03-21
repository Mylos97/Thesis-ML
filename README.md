# Thesis-ML
- We need some test data for checking if our implementations are correct
- Need to make training loops
- Maybe use Bayesian optimization for latent space
- Thompson sampling used by LERO and BAO
- Research multi armed problem for the operator specific model
- Improvements to Bayesian Optimization https://proceedings.neurips.cc/paper_files/paper/2022/file/ded98d28f82342a39f371c013dfb3058-Paper-Conference.pdf

- How should we prepare the input for the auto encoder model
- How do we prepare the data for the ONNX such that we can use it apache wayange
- How do prepare training data
- Implement Prepare trees in java :(
- Binary cross entropy 
- Soft max layer in decoder
- Remember to look into batch size scheduling

JAVA -> Encoding -> Onnx -> Result


# Usage

**Model names:**
- autoencoder (default value)
- pairwise
- treeconv

Example use model.py --model <model_name>
