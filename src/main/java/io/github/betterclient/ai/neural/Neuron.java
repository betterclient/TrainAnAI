package io.github.betterclient.ai.neural;

import java.util.HashMap;
import java.util.Map;

public class Neuron {
    public Map<Neuron, Double> connectionWeights = new HashMap<>();
    public Map<Neuron, Double> costGradientW = new HashMap<>();
    public double bias;
    public double costGradientB;

    public Neuron(double bias) {
        this.bias = bias;
    }

    public void initConnections(NeuralLayer layer) {
        for (Neuron neuron : layer.neurons) {
            connectionWeights.put(neuron, 0D);
        }
    }
}
