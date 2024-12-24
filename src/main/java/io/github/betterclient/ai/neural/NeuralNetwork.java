package io.github.betterclient.ai.neural;

import io.github.betterclient.ai.training.NetworkTrainer;
import io.github.betterclient.ai.training.TrainingInput;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    public List<NeuralLayer> layers = new ArrayList<>();

    public NeuralNetwork(int[] layerCounts) {
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

    public double learn(List<TrainingInput> data, float h, float learnRate) {
        List<NeuralLayer> layers = new ArrayList<>(this.layers);
        layers.removeFirst();
        for (NeuralLayer neuralLayer : layers) {
            float originalCost = NetworkTrainer.getCost(this, data);
            neuralLayer.learn(this, data, h, originalCost, learnRate);
        }

        return NetworkTrainer.getCost(this, data);
    }

    public float[] forward(float[] inputs) {
        for (int i = 1; i < layers.size(); i++) {
            inputs = layers.get(i).forward(inputs);
        }
        return inputs;
    }
}