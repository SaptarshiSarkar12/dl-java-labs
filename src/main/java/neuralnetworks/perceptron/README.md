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
Example usage and tests for the McCulloch-Pitts neuron can be found in the [MPNeuronTest.java](../../../../test/java/neuralnetworks/perceptron/MPNeuronTest.java) file.

## References

- ["A Logical Calculus of the Ideas Immanent in Nervous Activity" by Warren McCulloch and Walter Pitts (1943) - Wikipedia](https://en.wikipedia.org/wiki/A_Logical_Calculus_of_the_Ideas_Immanent_in_Nervous_Activity)
- [McCulloch-Pitts Neuron - Wikipedia](https://en.wikipedia.org/w/index.php?title=McCulloch_Pitts_neurons)