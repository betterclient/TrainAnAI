package io.github.betterclient.ai.neural;

import java.util.HashMap;
import java.util.Map;

public class Neuron {
    public Map<Neuron, Float> connectionWeights = new HashMap<>();
    public Map<Neuron, Float> costGradientW = new HashMap<>();
    public float bias;
    public float costGradientB;

    public Neuron(float bias) {
        this.bias = bias;
    }

    public void initConnections(NeuralLayer layer) {
        for (Neuron neuron : layer.neurons) {
            connectionWeights.put(neuron, 0F);
        }
    }
}
