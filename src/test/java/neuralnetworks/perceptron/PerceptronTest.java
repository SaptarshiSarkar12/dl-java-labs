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
    private final float learningRate = 0.1f;

    @Nested
    @DisplayName("Logic Gate Tests")
    class LogicGateTests {
        @Test
        @DisplayName("Perceptron learns 3-input AND gate")
        void testANDGateTraining() {
            float[][] inputs = threeInputTruthTable();
            int[] outputs = {0, 0, 0, 0, 0, 0, 0, 1}; // Only (1, 1, 1) should output 1 for AND gate

            Perceptron perceptron = new Perceptron(inputs, outputs, learningRate);
            perceptron.train(100);

            for (int i = 0; i < inputs.length; i++) {
                int actualOutput = perceptron.predict(inputs[i]);
                Assertions.assertEquals(outputs[i], actualOutput, "Failed for input " + Arrays.toString(inputs[i]));
            }
        }

        @Test
        @DisplayName("Perceptron learns 3-input OR gate")
        void testORGateTraining() {
            float[][] inputs = threeInputTruthTable();
            int[] outputs = {0, 1, 1, 1, 1, 1, 1, 1}; // Only (0, 0, 0) should output 0 for OR gate

            Perceptron perceptron = new Perceptron(inputs, outputs, learningRate);
            perceptron.train(100);

            for (int i = 0; i < inputs.length; i++) {
                int actualOutput = perceptron.predict(inputs[i]);
                Assertions.assertEquals(outputs[i], actualOutput, "Failed for input " + Arrays.toString(inputs[i]));
            }
        }

        @Test
        @DisplayName("Perceptron fails to learn XOR gate")
        void testXORGateFailure() {
            // XOR is not linearly separable. The Perceptron MUST fail.
            float[][] inputs = {
                    {1, 0, 0}, {1, 0, 1},
                    {1, 1, 0}, {1, 1, 1}
            };
            int[] outputs = {0, 1, 1, 0}; // XOR Logic

            Perceptron perceptron = new Perceptron(inputs, outputs, learningRate);
            // Train for a limited time
            boolean isConverged = perceptron.train(100);
            Assertions.assertFalse(isConverged, "Perceptron should not converge on XOR gate");
        }

        private float[][] threeInputTruthTable() {
            return new float[][] {
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
            float[][] trainingInputs = new float[trainingSize][3]; // [Bias, X, Y]
            int[] trainingOutputs = new int[trainingSize];

            Random rand = new Random(RANDOM_SEED);

            for (int i = 0; i < trainingSize; i++) {
                trainingInputs[i][0] = 1; // Bias input
                // Random x and y between 0 and 100
                float x = rand.nextFloat() * 100; // 0 to 100
                float y = rand.nextFloat() * 100; // 0 to 100

                trainingInputs[i][1] = x;
                trainingInputs[i][2] = y;

                // Define a linear decision boundary: (2*x + 3*y - 150) >= 0 => output = 1, else output = 0
                float logicValue = 2 * x + 3 * y - 150;
                trainingOutputs[i] = logicValue >= 0 ? 1 : 0;
            }

            Perceptron perceptron = new Perceptron(trainingInputs, trainingOutputs, learningRate);
            perceptron.train(500);

            // Test the trained perceptron on a random input
            float[][] testInputs = {
                    {1.0f, 10.0f, 10.0f}, // Should be 0
                    {1.0f, 50.0f, 50.0f}, // Should be 1
                    {1.0f, 20.0f, 30.0f}, // Should be 0
                    {1.0f, 5.0f, 5.0f}, // Should be 0
                    {1.0f, 70.0f, 3.0f}, // Should be 0 (close to boundary)
                    {1.0f, 0.0f, 60.0f} // Should be 1
            };
            int[] expectedOutputs = {0, 1, 0, 0, 0, 1};
            for (int i = 0; i < testInputs.length; i++) {
                int predictedOutput = perceptron.predict(testInputs[i]);
                Assertions.assertEquals(expectedOutputs[i], predictedOutput, "Failed for test input " + Arrays.toString(testInputs[i]));
            }
        }

        @Test
        @DisplayName("Constructor throws exception for mismatched input/output sizes")
        void testMismatchedInputOutputSizes() {
            float[][] inputs = {
                    {1, 0, 0}, {1, 0, 1},
                    {1, 1, 0}, {1, 1, 1}
            };
            int[] outputs = {0, 0}; // Incorrect size

            Assertions.assertThrows(IllegalArgumentException.class, () -> new Perceptron(inputs, outputs, learningRate), "Expected constructor to throw IllegalArgumentException due to mismatched input and output sizes");
        }

        @Test
        @DisplayName("Single sample training")
        void testSingleSample() {
            float[][] inputs = {{1, 1, 1}};
            int[] outputs = {1};

            Perceptron perceptron = new Perceptron(inputs, outputs, learningRate);
            perceptron.train(10);

            int result = perceptron.predict(inputs[0]);
            Assertions.assertEquals(1, result);
        }
    }
}
