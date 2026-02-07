package neuralnetworks.perceptron;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

@DisplayName("Single Layer Perceptron Implementation Tests")
public class PerceptronTest {
    // Use a fixed random seed for reproducibility in tests that involve random data generation
    private static final long RANDOM_SEED = 42L;
    private final double learningRate = 0.1;

    @Nested
    @DisplayName("Logic Gate Tests")
    class LogicGateTests {
        @Test
        @DisplayName("Perceptron learns 3-input AND gate")
        void testANDGateTraining() {
            double[][] inputs = threeInputTruthTable();
            int[] outputs = {0, 0, 0, 0, 0, 0, 0, 1}; // Only (1, 1, 1) should output 1 for AND gate

            Perceptron perceptron = new Perceptron(inputs, outputs, learningRate);
            perceptron.train(100);

            for (int i = 0; i < inputs.length; i++) {
                double actualOutput = perceptron.predict(inputs[i]);
                Assertions.assertEquals(outputs[i], actualOutput, "Failed for input " + Arrays.toString(inputs[i]));
            }
        }

        @Test
        @DisplayName("Perceptron learns 3-input OR gate")
        void testORGateTraining() {
            double[][] inputs = threeInputTruthTable();
            int[] outputs = {0, 1, 1, 1, 1, 1, 1, 1}; // Only (0, 0, 0) should output 0 for OR gate

            Perceptron perceptron = new Perceptron(inputs, outputs, learningRate);
            perceptron.train(100);

            for (int i = 0; i < inputs.length; i++) {
                double actualOutput = perceptron.predict(inputs[i]);
                Assertions.assertEquals(outputs[i], actualOutput, "Failed for input " + Arrays.toString(inputs[i]));
            }
        }

        @Test
        @DisplayName("Perceptron fails to learn XOR gate")
        void testXORGateFailure() {
            // XOR is not linearly separable. The Perceptron MUST fail.
            double[][] inputs = {
                    {1.0, 0.0, 0.0}, {1.0, 0.0, 1.0},
                    {1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}
            };
            int[] outputs = {0, 1, 1, 0}; // XOR Logic

            Perceptron perceptron = new Perceptron(inputs, outputs, learningRate);
            // Train for a limited time
            boolean isConverged = perceptron.train(100);
            Assertions.assertFalse(isConverged, "Perceptron should not converge on XOR gate");
        }

        private double[][] threeInputTruthTable() {
            return new double[][] {
                    {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
                    {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}
            };
        }
    }

    @Nested
    @DisplayName("General Behavior Tests")
    class GeneralBehaviorTests {
        @Test
        @DisplayName("Learns random linearly separable dataset")
        void testRandomLinearlySeparableData() {
            int trainingSize = 1000;
            double[][] trainingInputs = new double[trainingSize][3]; // [Bias, X, Y]
            int[] trainingOutputs = new int[trainingSize];

            Random rand = new Random(RANDOM_SEED);

            for (int i = 0; i < trainingSize; i++) {
                trainingInputs[i][0] = 1.0; // Bias input
                // Random x and y between 0 and 100
                double x = rand.nextDouble() * 100; // 0 to 100
                double y = rand.nextDouble() * 100; // 0 to 100

                trainingInputs[i][1] = x;
                trainingInputs[i][2] = y;

                // Define a linear decision boundary: (2*x + 3*y - 150) >= 0 => output = 1, else output = 0
                double logicValue = 2 * x + 3 * y - 150;
                trainingOutputs[i] = logicValue >= 0 ? 1 : 0;
            }

            Perceptron perceptron = new Perceptron(trainingInputs, trainingOutputs, learningRate);
            perceptron.train(500);

            // Test the trained perceptron on a random input
            double[][] testInputs = {
                    {1.0, 10.0, 10.0}, // Should be 0
                    {1.0, 50.0, 50.0}, // Should be 1
                    {1.0, 20.0, 30.0}, // Should be 0
                    {1.0, 5.0, 5.0}, // Should be 0
                    {1.0, 70.0, 3.0}, // Should be 0 (close to boundary)
                    {1.0, 0.0, 60.0} // Should be 1
            };
            int[] expectedOutputs = {0, 1, 0, 0, 0, 1};
            for (int i = 0; i < testInputs.length; i++) {
                double predictedOutput = perceptron.predict(testInputs[i]);
                Assertions.assertEquals(expectedOutputs[i], predictedOutput, "Failed for test input " + Arrays.toString(testInputs[i]));
            }
        }

        @Test
        @DisplayName("Constructor throws exception for mismatched input/output sizes")
        void testMismatchedInputOutputSizes() {
            double[][] inputs = {
                    {1, 0, 0}, {1, 0, 1},
                    {1, 1, 0}, {1, 1, 1}
            };
            int[] outputs = {0, 0}; // Incorrect size

            Assertions.assertThrows(IllegalArgumentException.class, () -> new Perceptron(inputs, outputs, learningRate), "Expected constructor to throw IllegalArgumentException due to mismatched input and output sizes");
        }

        @Test
        @DisplayName("Single sample training")
        void testSingleSample() {
            double[][] inputs = {{1, 1, 1}};
            int[] outputs = {1};

            Perceptron perceptron = new Perceptron(inputs, outputs, learningRate);
            perceptron.train(10);

            int result = perceptron.predict(inputs[0]);
            Assertions.assertEquals(1, result);
        }
    }
}
