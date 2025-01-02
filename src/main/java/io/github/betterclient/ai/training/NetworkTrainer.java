package io.github.betterclient.ai.training;

import io.github.betterclient.ai.neural.NeuralLayer;
import io.github.betterclient.ai.neural.NeuralNetwork;
import io.github.betterclient.ai.neural.Neuron;

import java.util.List;

public class NetworkTrainer {
    public static float train(NeuralNetwork network, List<TrainingInput> samples, float learnRate) {
        //Slower, more accurate training for smaller models.
        if (samples.size() < 1000 && network.getParameterSize() < 1000) {
            SmallNetworkTrainer.train(network, samples, learnRate, learnRate);
            return 1;
        }

        float or = network.cost(samples);

        for (NeuralLayer layer : network.layersW()) {
            for (Neuron neuronIn : layer.before.neurons) {
                for (Neuron neuronOut : layer.neurons) {
                    neuronIn.connectionWeights.put(neuronOut, neuronIn.connectionWeights.get(neuronOut) + learnRate);
                    float costDelta = network.cost(samples) - or;
                    neuronIn.connectionWeights.put(neuronOut, neuronIn.connectionWeights.get(neuronOut) - learnRate);
                    neuronIn.connectionWeightCosts.put(neuronOut, costDelta / learnRate);
                }
            }

            for (Neuron neuronIn : layer.neurons) {
                neuronIn.bias += learnRate;
                float costDelta = network.cost(samples) - or;
                neuronIn.bias -= learnRate;
                neuronIn.biasCost = costDelta / learnRate;
            }
        }

        for (NeuralLayer layer : network.layersW()) {
            layer.applyGradients(learnRate);
        }

        //return network.cost(samples);
        return 0;
    }
}