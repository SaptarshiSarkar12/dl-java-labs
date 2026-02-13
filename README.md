# Deep Learning From Scratch (Java-first)

[![Test](https://github.com/SaptarshiSarkar12/dl-java-labs/actions/workflows/test.yml/badge.svg)](https://github.com/SaptarshiSarkar12/dl-java-labs/actions/workflows/test.yml)

<img width="1536" height="1024" alt="Project Banner" src="https://github.com/user-attachments/assets/e8c8b6f9-51ea-4eda-8f73-05e90b3eb0b5" />

This repository documents my journey of learning deep learning from first principles through:
- Concept breakdowns
- Mathematical intuition
- From-scratch implementations (mostly in Java)

Each section is aligned with a LinkedIn post where I summarize insights and lessons learned.

## Motivation
The goal of this repository is to develop a deep, first-principles understanding of deep learning rather than relying on high-level abstractions.

## Why Java?
To understand what happens under the hood by implementing core concepts manually instead of relying on high-level, black-box libraries.

## Repository Organization

- [Fundamentals](./src/main/java/fundamentals) – Mathematical foundations.
    - [Linear Algebra](./src/main/java/fundamentals/linearalgebra) – Vector and matrix operations.
- [Neural Networks](./src/main/java/neuralnetworks) – Core neural network models.
    - [Perceptron](./src/main/java/neuralnetworks/perceptron) – Basic building blocks of neural networks.
        - [McCulloch–Pitts Neuron](./src/main/java/neuralnetworks/perceptron/MPNeuron.java) – Binary threshold neuron with inhibitory inputs.
        - [Perceptron](./src/main/java/neuralnetworks/perceptron/Perceptron.java) – Extension of MPNeuron with learnable weights and bias.
        - [Multi-layer Perceptron](./src/main/java/neuralnetworks/perceptron/MLP.java) – A simple feedforward neural network with one hidden layer.

## Running Tests

This project uses JUnit for testing.

### Requirements
- Java 25 (Oracle JDK 25)
- Maven

### Run all tests
From the root of the repository, run:
```bash
mvn test
```

## Learning Sources
- MIT 18.06 Linear Algebra, Spring 2005 (Lecture Videos) by **Gilbert Strang** (YouTube)
- "Neural Networks and Deep Learning" by Michael Nielsen (neuralnetworksanddeeplearning.com)
- "Deep Learning" playlist (YouTube)
- Additional articles, papers, and references as needed

## Status
🚧 Actively learning and building.  
This repository will be updated incrementally as concepts are implemented.