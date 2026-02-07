# Deep Learning From Scratch (Java-first)

[![Test](https://github.com/SaptarshiSarkar12/dl-java-labs/actions/workflows/test.yml/badge.svg)](https://github.com/SaptarshiSarkar12/dl-java-labs/actions/workflows/test.yml)

<img width="1536" height="1024" alt="Project Banner" src="https://github.com/user-attachments/assets/e8c8b6f9-51ea-4eda-8f73-05e90b3eb0b5" />

This repository documents my deep learning journey through:
- Concept breakdowns
- Mathematical intuition
- From-scratch implementations (mostly in Java)

Each section is aligned with a LinkedIn post where I summarize insights and lessons learned.

## Motivation
The goal of this repository is to develop a deep, first-principles understanding of deep learning rather than relying on high-level abstractions.

## Why Java?
To deeply understand what happens under the hood instead of relying on black-box libraries.

## Repository Organization

- [Fundamentals](./src/main/java/fundamentals) - Basic concepts and mathematical foundations.
  - [Linear Algebra](./src/main/java/fundamentals/linearalgebra) - Vectors and Matrices Operations
- [Neural Networks](./src/main/java/neuralnetworks) - Building blocks of neural networks.
    - [Perceptron](./src/main/java/neuralnetworks/perceptron) - Single-layer neural network implementation.
        - [McCulloch-Pitts Neuron](./src/main/java/neuralnetworks/perceptron/MPNeuron.java) - Basic binary classifier without weights or bias.
        - [Perceptron](./src/main/java/neuralnetworks/perceptron/Perceptron.java) - Weighted binary classifier with bias.

## Learning Sources
- MIT 18.06 Linear Algebra, Spring 2005 (Lecture Videos) by **Gilbert Strang** (YouTube)
- "Neural Networks and Deep Learning" by Michael Nielsen (neuralnetworksanddeeplearning.com)
- "Deep Learning" playlist (YouTube)
- Additional articles, papers, and references as needed

## Status
🚧 Actively learning and building.  
This repository will be updated incrementally as concepts are implemented.