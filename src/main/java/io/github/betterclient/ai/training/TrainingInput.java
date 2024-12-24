package io.github.betterclient.ai.training;

import java.util.Arrays;

public class TrainingInput {
    public float[] inputs;
    /**
     * Expected results for input
     */
    public float[] expected;

    public TrainingInput(float[] inputs, float[] expected) {
        this.inputs = inputs;
        this.expected = expected;
    }

    @Override
    public String toString() {
        return "Input{" + Arrays.toString(inputs) + ": " + Arrays.toString(expected) + "}";
    }
}
