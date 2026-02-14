package neuralnetworks.perceptron;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import utils.SimdMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.stream.IntStream;

public class Perceptron {
    private static final Logger logger = LoggerFactory.getLogger(Perceptron.class);
    private final float[][] inputs; // input[][0] is bias input (always 1)
    private final int[] outputs; // output values: 0 or 1
    private SimdMatrix weights; // w[0] is bias weight (= -threshold)
    private final float learningRate;

    public Perceptron(float[][] inputs, int[] outputs, float learningRate) {
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException("Number of input samples must match number of output samples.");
        }
        this.inputs = inputs;
        this.outputs = outputs;
        this.learningRate = learningRate;
        initializeWeights();
    }

    public boolean train(int maxEpochs) {
        int columns = inputs[0].length; // Number of features (including bias)
        int epoch = 0;
        boolean converged = false;
        // A list of indices (0 to inputs.length-1) that we will shuffle each epoch for random order training
        ArrayList<Integer> indices = new ArrayList<>(IntStream.range(0, inputs.length).boxed().toList());

        logger.info("Starting training with learning rate: {}, max epochs: {}", learningRate, maxEpochs);
        while (!converged && epoch < maxEpochs) {
            converged = true;
            Collections.shuffle(indices); // Shuffle indices to ensure random order of training samples each epoch

            for (int i : indices) {
                SimdMatrix xVector = new SimdMatrix(1, columns, inputs[i]);
                int y = outputs[i];

                double dotProduct = getDotProduct(xVector, weights);
                int predicted = stepFunction(dotProduct);

                // Standard update rule: w = w + learningRate * (y - predicted) * x
                // If y == predicted, no update needed.
                // If y == 1 and predicted == 0, we need to add x to weights (multiplier = +1)
                // If y == 0 and predicted == 1, we need to subtract x from weights (multiplier = -1)
                int error = y - predicted;
                if (error != 0) {
                    updateWeights(weights, xVector, error); // error is +1 or -1, so it will add or subtract x from weights
                    converged = false; // If we had to update weights, we are not yet converged
                    logger.debug("Epoch {}: Update triggered. Input: {}, Error: {}", epoch, xVector, error);
                }
            }
            epoch++;
        }
        if (converged) {
            logger.info("Training converged successfully after {} epochs.", epoch - 2); // Subtract 2 because we increment epoch at the end of the loop, so it will be 1 more than the actual last epoch where we had convergence
            logger.debug("Trained weights: {}", weights);
        } else {
            logger.warn("Training failed to converge after {} epochs.", maxEpochs);
        }
        return converged;
    }

    public int predict(float[] input) {
        if (input.length != weights.columns()) {
            throw new IllegalArgumentException("Input size does not match weight size.");
        }
        SimdMatrix inputVec = new SimdMatrix(1, input.length, input);
        double dotProduct = getDotProduct(inputVec, weights);
        return stepFunction(dotProduct);
    }

    private int stepFunction(double dotProduct) {
        if (dotProduct >= 0) {
            return 1;
        } else {
            return 0;
        }
    }

    private void updateWeights(SimdMatrix weights, SimdMatrix x, int multiplier) {
        // multiplier is +1 if we need to add x to weights, -1 if we need to subtract x from weights
        this.weights = weights.addRowVector(x.scale(learningRate * multiplier));
    }

    public float getDotProduct(SimdMatrix inputs, SimdMatrix weights) {
        if (inputs.columns() != weights.columns()) {
            throw new IllegalArgumentException("Input size does not match weight size.");
        }
        return weights.elementMult(inputs).sum();
    }

    public void initializeWeights() {
        this.weights = SimdMatrix.random(1, inputs[0].length);
        logger.debug("Weights initialized to: {}", weights);
    }
}
