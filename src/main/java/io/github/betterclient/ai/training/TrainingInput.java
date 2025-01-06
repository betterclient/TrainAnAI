package io.github.betterclient.ai.training;

import java.util.Arrays;

public class TrainingInput {
    public double[] inputs;
    /**
     * Expected results for input
     */
    public double[] expected;

    public TrainingInput(double[] inputs, double[] expected) {
        this.inputs = inputs;
        this.expected = expected;
    }

    @Override
    public String toString() {
        return "Input{" + Arrays.toString(inputs) + ": " + Arrays.toString(expected) + "}";
    }
}
