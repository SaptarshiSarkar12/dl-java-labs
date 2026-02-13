## Utilities Package

This package contains helper classes and functions that support the core implementations of neural networks. It includes utilities for data manipulation, activation functions, and optimized matrix operations.

### [SIMD Matrix Operations](SimdMatrix.java)

The `SimdMatrix` class provides optimized matrix operations using SIMD (Single Instruction, Multiple Data) instructions. This allows for improved performance when performing matrix computations, which are fundamental to neural network training and inference.

#### Key Features:
- **Matrix Multiplication**: Efficiently multiplies two matrices using SIMD instructions.
- **Element-wise Operations**: Supports element-wise addition, subtraction, and other operations optimized for performance.

> **Note**: Ensure that your system supports SIMD instructions to take full advantage of the performance benefits provided by this class.