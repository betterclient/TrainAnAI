package io.github.betterclient.ai.training;

import io.github.betterclient.ai.neural.NeuralLayer;
import io.github.betterclient.ai.neural.NeuralNetwork;

import java.util.List;

public class NetworkTrainer {
    public static float train(NeuralNetwork network, List<TrainingInput> samples, float learnRate) {
        float startingCost = network.cost(samples);

        for (TrainingInput sample : samples) {
            network.updateAllGradients(sample);
        }

        for (NeuralLayer layer : network.layersW()) {
            //learnRate delta: 739.0078
            //learnRate / samples.size() delta: 39.164062
            //learnRate * samples.size() delta: 626.3164

            layer.applyGradients(learnRate);
        }

        for (NeuralLayer layer : network.layersW()) {
            layer.clearGradients();
        }

        return startingCost - network.cost(samples);
    }
}