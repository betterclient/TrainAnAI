package io.github.betterclient.ai.training;

import io.github.betterclient.ai.neural.NeuralLayer;
import io.github.betterclient.ai.neural.NeuralNetwork;

import java.util.List;

public class NetworkTrainer {
    public static float train(NeuralNetwork network, List<TrainingInput> samples, float learnRate) {
        //Slower, more accurate training for smaller models.
        if (samples.size() < 1000 && network.getParameterSize() < 1000) {
            SmallNetworkTrainer.train(network, samples, learnRate, learnRate);
            return 1;
        }

        for (TrainingInput sample : samples) {
            network.updateAllGradients(sample);
        }

        for (NeuralLayer layer : network.layersW()) {
            //May change for different models
            //learnRate delta: 739.0078
            //learnRate / samples.size() delta: 39.164062
            //learnRate * samples.size() delta: 626.3164

            layer.applyGradients(learnRate);
        }

        for (NeuralLayer layer : network.layersW()) {
            layer.clearGradients();
        }

        //return network.cost(samples);
        return 0;
    }
}