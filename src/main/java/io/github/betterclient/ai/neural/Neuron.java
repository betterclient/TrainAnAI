package io.github.betterclient.ai.neural;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

public class Neuron {
    public Map<Neuron, BigDecimal> connectionWeights = new HashMap<>();
    public double bias;

    public double lastDerivative;
    public double error;
    public double lastOutput;

    public Neuron(float bias) {
        this.bias = bias;
    }

    public void initConnections(NeuralLayer layer) {
        for (Neuron neuron : layer.neurons) {
            connectionWeights.put(neuron, BigDecimal.valueOf(Math.random() - 0.5));
        }
    }
}