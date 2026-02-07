package neuralnetworks.perceptron;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.stream.IntStream;

public class Perceptron {
    private static final Logger logger = LoggerFactory.getLogger(Perceptron.class);
    private final double[][] inputs; // input[][0] is bias input (always 1)
    private final int[] outputs; // output values: 0 or 1
    private double[] weights; // w[0] is bias weight (= -threshold)
    private final double learningRate;

    public Perceptron(double[][] inputs, int[] outputs, double learningRate) {
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException("Number of input samples must match number of output samples.");
        }
        this.inputs = inputs;
        this.outputs = outputs;
        this.learningRate = learningRate;
        initializeWeights();
    }

    public boolean train(int maxEpochs) {
        int epoch = 0;
        boolean converged = false;
        // A list of indices (0 to inputs.length-1) that we will shuffle each epoch for random order training
        ArrayList<Integer> indices = new ArrayList<>(IntStream.range(0, inputs.length).boxed().toList());

        logger.info("Starting training with learning rate: {}, max epochs: {}", learningRate, maxEpochs);
        while (!converged && epoch < maxEpochs) {
            converged = true;
            Collections.shuffle(indices); // Shuffle indices to ensure random order of training samples each epoch

            for (int i : indices) {
                double[] x = inputs[i];
                int y = outputs[i];

                double dotProduct = getDotProduct(x, weights);
                int predicted = stepFunction(dotProduct);

                // Standard update rule: w = w + learningRate * (y - predicted) * x
                // If y == predicted, no update needed.
                // If y == 1 and predicted == 0, we need to add x to weights (multiplier = +1)
                // If y == 0 and predicted == 1, we need to subtract x from weights (multiplier = -1)
                int error = y - predicted;
                if (error != 0) {
                    updateWeights(weights, x, error); // error is +1 or -1, so it will add or subtract x from weights
                    converged = false; // If we had to update weights, we are not yet converged
                    logger.debug("Epoch {}: Update triggered. Input: {}, Error: {}", epoch, Arrays.toString(x), error);
                }
            }
            epoch++;

        }
        if (converged) {
            logger.info("Training converged successfully after {} epochs.", epoch);
            logger.debug("Trained weights: {}", Arrays.toString(weights));
        } else {
            logger.warn("Training failed to converge after {} epochs.", epoch);
        }
        return converged;
    }

    public int predict(double[] input) {
        if (input.length != weights.length) {
            throw new IllegalArgumentException("Input size does not match weight size.");
        }
        double dotProduct = getDotProduct(input, weights);
        return stepFunction(dotProduct);
    }

    private int stepFunction(double dotProduct) {
        if (dotProduct >= 0) {
            return 1;
        } else {
            return 0;
        }
    }

    private void updateWeights(double[] weights, double[] x, int multiplier) {
        // multiplier is +1 if we need to add x to weights, -1 if we need to subtract x from weights
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * multiplier * x[i];
        }
    }

    public double getDotProduct(double[] inputs, double[] weights) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException("Input size does not match weight size.");
        }
        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return sum;
    }

    public void initializeWeights() {
        this.weights = new double[inputs[0].length];
        Random rand = new Random();
        // Initialize weights to symmetric random values between -0.01 and 0.01 (small random values around zero)
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (rand.nextDouble() * 0.02) - 0.01; // Random value in range [-0.01, 0.01]
        }
        logger.debug("Weights initialized to: {}", Arrays.toString(weights));
    }
}
