package neuralnetworks.perceptron;

public class MPNeuron {
    boolean[] inputs; // input values: 0 or 1
    boolean[] inhibitory; // true if inhibitory, false if excitatory
    int threshold;

    public MPNeuron(boolean[] inputs, boolean[] inhibitory, int threshold) {
        this.inputs = inputs;
        this.inhibitory = inhibitory;
        this.threshold = threshold;
    }

    public int output() {
        if (hasActiveInhibitoryInput()) {
            return 0;
        }
        return excitatorySum() >= threshold ? 1 : 0;
    }

    private boolean hasActiveInhibitoryInput() {
        for (int i = 0; i < inputs.length; i++) {
            if (inhibitory[i] && inputs[i]) {
                return true;
            }
        }
        return false;
    }

    private int excitatorySum() {
        int sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            if (!inhibitory[i] && inputs[i]) {
                sum++;
            }
        }
        return sum;
    }
}
