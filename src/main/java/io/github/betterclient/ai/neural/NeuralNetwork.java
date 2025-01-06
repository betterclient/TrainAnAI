package io.github.betterclient.ai.neural;

import io.github.betterclient.ai.training.TrainingInput;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    public List<NeuralLayer> layers = new ArrayList<>();
    public final int[] layerCounts;

    public NeuralNetwork(int[] layerCounts) {
        this.layerCounts = layerCounts;
        for (int layerCount : layerCounts) {
            layers.add(new NeuralLayer(layerCount));
        }

        NeuralLayer before = null;
        int index = 0;
        for (NeuralLayer layer : layers) {
            index++;

            layer.before = before;
            before = layer;

            if (layer != layers.getLast()) {
                layer.after = layers.get(index);
            }
        }

        for (NeuralLayer layer : layers) {
            if (layer != layers.getLast()) {
                for (Neuron neuron : layer.neurons) {
                    neuron.initConnections(layers.get(layers.indexOf(layer) + 1));
                }
            }
        }
    }

    public double[] forward(double[] inputs) {
        for (int i = 1; i < layers.size(); i++) {
            NeuralLayer layer = layers.get(i);

            inputs = layer.forward(inputs);
        }
        return inputs;
    }

    public double cost(List<TrainingInput> samples) {
        double cost = 0;
        for (TrainingInput data : samples) {
            cost += cost(data);
        }
        return cost;
    }

    public double cost(TrainingInput sample) {
        double cost = 0;
        double[] actualOutput = this.forward(sample.inputs);

        for (int i = 0; i < actualOutput.length; i++) {
            double error = actualOutput[i] - sample.expected[i];
            cost += error * error;
        }
        return cost;
    }

    public int getParameterSize() {
        int totalParameters = 0;
        for (int i = 0; i < layerCounts.length - 1; i++) {
            int currentLayerSize = layerCounts[i];
            int nextLayerSize = layerCounts[i + 1];
            totalParameters += currentLayerSize * nextLayerSize;
            totalParameters += nextLayerSize;
        }
        return totalParameters;
    }
}