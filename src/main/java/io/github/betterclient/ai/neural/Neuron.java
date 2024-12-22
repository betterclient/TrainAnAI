package io.github.betterclient.ai.neural;

import java.util.HashMap;
import java.util.Map;

public class Neuron {
    public Map<Neuron, Double> connectionWeights = new HashMap<>();
    public double bias;

    public Neuron(double bias) {
        this.bias = bias;
    }

    public void initConnections(NeuralLayer layer) {
        for (Neuron neuron : layer.neurons) {
            connectionWeights.put(neuron, Math.random() - 0.5);
        }
    }
}
