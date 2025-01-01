package io.github.betterclient.ai.training;

import io.github.betterclient.ai.neural.NeuralLayer;
import io.github.betterclient.ai.neural.NeuralNetwork;
import io.github.betterclient.ai.neural.Neuron;

import java.util.ArrayList;
import java.util.List;

/**
 * More accurate training for smaller models
 */
public class SmallNetworkTrainer {
    public static void train(NeuralNetwork network, List<TrainingInput> samples, float h, float learningRate) {
        List<NeuralLayer> layers = new ArrayList<>(network.layers);
        layers.removeFirst();
        for (NeuralLayer neuralLayer : layers) {
            float originalCost = network.cost(samples);
            learn(neuralLayer, network, samples, h, originalCost, learningRate);
        }
    }

    public static void learn(NeuralLayer layer, NeuralNetwork network, List<TrainingInput> samples, float h, float currentCost, float learningRate) {
        for (Neuron neuronIn : layer.before.neurons) {
            for (Neuron neuronOut : layer.neurons) {
                float value = neuronIn.connectionWeights.get(neuronOut);
                neuronIn.connectionWeights.put(neuronOut, value + h);

                float cost = network.cost(samples);
                float delta = cost - currentCost;

                if (delta > 0) {
                    //Wrong direction
                    value -= learningRate;
                } else {
                    //Correct direction
                    value += learningRate;
                }

                neuronIn.connectionWeights.put(neuronOut, value);

                currentCost = network.cost(samples);
            }
        }

        currentCost =  network.cost(samples);

        for (Neuron neuron : layer.neurons) {
            neuron.bias += h;
            float currentCost0 = network.cost(samples);
            float delta = (currentCost0 - currentCost);
            if (delta != 0) {
                //Change happened

                if (delta > 0) {
                    //Wrong direction
                    neuron.bias -= learningRate;
                } else {
                    //Correct direction
                    neuron.bias += learningRate;
                }
            }
            neuron.bias -= h;
            currentCost = currentCost0;
        }
    }
}