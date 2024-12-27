package io.github.betterclient.ai.training;

import io.github.betterclient.ai.neural.NeuralLayer;

public class PreCalculation {
    public float[] inputs;
    public float[] weightedInputs;
    public float[] activations;
    public float[] nodeValues;

    public PreCalculation(NeuralLayer layer) {
        weightedInputs = new float[layer.neurons.size()];
        activations = new float[layer.neurons.size()];
        nodeValues = new float[layer.neurons.size()];
    }
}
