## McCulloch-Pitts neuron

### Intuition
The McCulloch-Pitts neuron is a simplified model of a biological neuron. It takes multiple binary inputs, aggregates them and produces a binary output based on whether the aggregated input exceeds a certain threshold.

### Mathematical Representation

The output `y` of a McCulloch-Pitts neuron can be represented mathematically as:

```
y = f(g(x_1, x_2, ..., x_n))
  = 0, if g(x_1, x_2, ..., x_n) < θ or any inhibitory input is active
    1, if g(x_1, x_2, ..., x_n) ≥ θ
```
Where:
- `x_1, x_2, ..., x_n` are the binary inputs (0 or 1).
- `θ` is the threshold.
- `g` is the aggregation function (typically summation).
- `f` is the activation function (step function).
- Inhibitory inputs are special inputs that can prevent the neuron from firing (outputting 1) regardless of the aggregated input.

### Implementation
Implementation of the McCulloch-Pitts neuron in Java can be found in the [MPNeuron.java](MPNeuron.java) file.

### Tests/Examples
Example usage and tests for the McCulloch-Pitts neuron can be found in the [MPNeuronTest.java](MPNeuronTest.java) file in `src/test/java/neuralnetworks/perceptron` directory.

---

## Single-layer Perceptron

### Intuition
A single-layer perceptron is a simple neural network that consists of a single layer of perceptrons. It can only learn to classify **linearly separable data**. The output of the single-layer perceptron is determined by the **weighted sum of the inputs** and the **bias**, passed through an **activation function** usually a _step function_.

The single-layer perceptron can only learn to classify data that is **linearly separable**, meaning that there exists a hyperplane that can separate the classes in the input space.

### Mathematical Representation
The output `y` of a single-layer perceptron can be represented mathematically as:

```y = f(w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b)```

Where:
- `x_1, x_2, ..., x_n` are the inputs.
- `w_1, w_2, ..., w_n` are the weights associated with each input.
- `b` is the bias term (negative threshold).
- `f` is the activation function (commonly a **step function**).

### Implementation
Implementation of the Single-layer Perceptron in Java can be found in the [Perceptron.java](Perceptron.java) file.

### Tests/Examples
Example usage and tests for the Single-layer Perceptron can be found in the [PerceptronTest.java](PerceptronTest.java) file in `src/test/java/neuralnetworks/perceptron` directory.

## References

- ["A Logical Calculus of the Ideas Immanent in Nervous Activity" by Warren McCulloch and Walter Pitts (1943) - Wikipedia](https://en.wikipedia.org/wiki/A_Logical_Calculus_of_the_Ideas_Immanent_in_Nervous_Activity)
- [McCulloch-Pitts Neuron - Wikipedia](https://en.wikipedia.org/w/index.php?title=McCulloch_Pitts_neurons)
- [Perceptron - Wikipedia](https://en.wikipedia.org/wiki/Perceptron)