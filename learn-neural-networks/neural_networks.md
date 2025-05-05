# How Neural Networks Work

This section explains the structure, functionality, and learning process of neural networks. It builds on the foundations from [Understanding Machine Learning and Neural Networks](machine_learning_overview.md) and prepares you for topics like [Language Models (LMs)](language_models.md).

## What is a Neural Network?

A neural network is a computational model composed of layers of interconnected nodes, or "neurons." Each neuron processes input data, applies a mathematical transformation, and passes the result to the next layer, ultimately producing an output.

### Basic Structure

- **Input Layer**: Receives the raw data (e.g., pixel values of an image or tokenized text).
- **Hidden Layers**: Perform computations to extract features and patterns.
- **Output Layer**: Produces the final result (e.g., a classification or generated text).

Each connection between neurons has a **weight**, which is adjusted during training to improve accuracy. A **bias** term is also added to each neuron to shift the output, enhancing flexibility.

## How Do Neural Networks Learn?

Neural networks learn through an iterative process involving **forward propagation**, **loss calculation**, **backpropagation**, and **gradient descent**.

1. **Forward Propagation**:
   - Data passes through the layers, with each neuron computing a weighted sum of inputs, adding a bias, and applying an **activation function** (e.g., ReLU, Sigmoid) to introduce non-linearity.
   - The output is compared to the expected result.

2. **Loss Calculation**:
   - A **loss function** measures the difference between the predicted and actual output (e.g., mean squared error for regression, cross-entropy for classification).
   - The goal is to minimize this loss.

3. **Backpropagation**:
   - The error is propagated backward through the network, calculating how each weight and bias contributes to the loss.
   - This uses the chain rule from calculus to compute gradients.

4. **Gradient Descent**:
   - An optimization algorithm adjusts weights and biases to reduce the loss.
   - The **learning rate** controls the size of these adjustments—too high causes instability, too low slows learning.

This process repeats over multiple **epochs** (full passes through the training data), with data processed in **batches** to balance speed and stability.

### Key Terms

- **Activation Function**: Determines whether a neuron activates (e.g., ReLU outputs the input if positive, else 0).
- **Epoch**: One complete pass through the training data.
- **Batch**: A subset of the training data used in one iteration.

## Types of Neural Networks

- **Feedforward Neural Networks (FNN)**: Basic type with data flowing in one direction, used for simple tasks.
- **Convolutional Neural Networks (CNN)**: Specialized for image data, using convolutional layers to detect features like edges.
- **Recurrent Neural Networks (RNN)**: Designed for sequential data like text, with loops to maintain memory of previous inputs.
- **Transformers**: Advanced architecture for NLP, using self-attention to process entire sequences simultaneously, as discussed in [Large Language Models (LLMs)](large_language_models.md).

The code in [Explanation of the Code](code_explanation.md) uses a Transformer-based model for language tasks.

## Why Neural Networks Matter

Neural networks power many modern AI applications, from voice assistants to autonomous vehicles. Their ability to learn complex patterns makes them ideal for tasks where traditional algorithms struggle.

## Next Steps

Explore [Language Models (LMs)](language_models.md) to see how neural networks are applied to text processing, or jump to [Libraries Used in the Code](libraries.md) to understand the tools for building them.

[Back to README](README.md)
