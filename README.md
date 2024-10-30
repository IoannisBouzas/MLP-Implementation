# Four-Category Classification using Multilayer Perceptron

## Project Overview
This project implements a Multilayer Perceptron (MLP) neural network to solve a four-category classification problem within a 2D space. The network is designed to classify points in the [-1,1]x[-1,1] square into four distinct categories based on specific conditions.

## Dataset Generation
- Total examples: 8000
- Training set: 4000 examples
- Control set: 4000 examples
- Input space: Points (x₁, x₂) in [-1,1]x[-1,1]
- Output: Four categories based on defined conditions

## Neural Network Architecture
### Network Structure
- Input layer: 2 neurons (x₁, x₂ coordinates)
- Hidden layers: 3 layers with configurable neurons
- Output layer: 4 neurons (one for each category)

### Key Components
1. **Input Processing**
   - Normalization of input data
   - Data encoding for categorical outputs

2. **Network Configuration**
   - Configurable number of neurons per hidden layer
   - Selectable activation functions
   - Adjustable batch size for training

3. **Training Process**
   - Forward propagation
   - Backpropagation
   - Gradient descent optimization
   - Dynamic learning rate
   - Early stopping based on error threshold

## Usage Guide

1. **Dataset Preparation**


2. **Network Configuration**


3. **Training**


4. **Evaluation**

## Hyperparameter Optimization
The network's performance can be optimized by tuning:
- Hidden layer sizes
- Activation functions (ReLU, tanh, sigmoid)
- Batch size
- Learning rate
- Error threshold

## Visualization
The project includes visualization tools to:
- Plot decision boundaries
- Display classification results
- Highlight correct/incorrect predictions
- Show training convergence

## Results Analysis
The program outputs:
1. Training error progression
2. Final generalization capacity
3. Visual representation of:
   - Correctly classified points
   - Misclassified points
   - Decision boundaries

## Best Practices
1. **Data Preprocessing**
   - Normalize input data
   - Shuffle training examples
   - Split data consistently

2. **Training**
   - Monitor convergence
   - Use appropriate batch sizes
   - Implement early stopping

3. **Evaluation**
   - Use separate control set
   - Calculate confusion matrix
   - Analyze error patterns


