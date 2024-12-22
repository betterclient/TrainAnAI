package io.github.betterclient.ai.neural;

import io.github.betterclient.ai.training.NetworkTrainer;
import io.github.betterclient.ai.training.TrainingDataPoint;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

    public double[] forward(double[] inputs) {
        for (int i = 1; i < layers.size(); i++) {
            inputs = layers.get(i).forward(inputs);
        }
        return inputs;
    }

    public TrainingDataPoint convert(double[][] inputs, double[][] expectedOutputs) {
        double cost = NetworkTrainer.getCost(this, inputs, expectedOutputs);
        List<TrainingDataPoint.NeuronPoint> points = convertPoints();

        return new TrainingDataPoint(cost, points);
    }

    private List<TrainingDataPoint.NeuronPoint> convertPoints() {
        List<TrainingDataPoint.NeuronPoint> points = new ArrayList<>();

        for (Neuron neuron : this.layers.getFirst().neurons) {
            TrainingDataPoint.NeuronPoint point = getPoint(neuron);
            points.add(point);
        }

        return points;
    }

    private static TrainingDataPoint.NeuronPoint getPoint(Neuron neuron) {
        Map<TrainingDataPoint.NeuronPoint, Double> weightConnections = new HashMap<>();
        TrainingDataPoint.NeuronPoint point = new TrainingDataPoint.NeuronPoint(neuron.bias, weightConnections);
        for (Map.Entry<Neuron, Double> weights : neuron.connectionWeights.entrySet()) {
            weightConnections.put(getPoint(weights.getKey()), weights.getValue());
        }
        return point;
    }
}
