package neuralnetworks.perceptron;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Tests for McCulloch-Pitts Neuron (MPNeuron)")
public class MPNeuronTest {
    @Test
    @DisplayName("Test AND gate behavior of MPNeuron")
    void testANDGateBehaviour() {
        boolean[] x = {true, true, true};
        boolean[] inhibitory = {false, false, false};
        MPNeuron neuron = new MPNeuron(x, inhibitory, 3);
        Assertions.assertEquals(1, neuron.output(), "AND gate should output 1 when all inputs are true");
    }

    @Test
    @DisplayName("Test inhibitory input behavior of MPNeuron")
    void testInhibitoryInputBehaviour() {
        boolean[] x = {true, false, true};
        boolean[] inhibitory = {true, false, false};
        MPNeuron neuron = new MPNeuron(x, inhibitory, 2);
        Assertions.assertEquals(0, neuron.output(), "Neuron should output 0 when there is an active inhibitory input");
    }

    @Test
    @DisplayName("Test threshold behavior of MPNeuron")
    void testThresholdBehaviour() {
        boolean[] x = {true, true, false};
        boolean[] inhibitory = {false, false, false};
        MPNeuron neuron = new MPNeuron(x, inhibitory, 3);
        Assertions.assertEquals(0, neuron.output(), "Neuron should output 0 when excitatory sum is below threshold");
    }
}
