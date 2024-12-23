package io.github.betterclient.ai.training;

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
}
