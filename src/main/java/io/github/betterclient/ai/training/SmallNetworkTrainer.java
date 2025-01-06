package io.github.betterclient.ai.training;

import io.github.betterclient.ai.neural.NeuralLayer;
import io.github.betterclient.ai.neural.NeuralNetwork;
import io.github.betterclient.ai.neural.Neuron;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * More accurate training for smaller models
 */
public class SmallNetworkTrainer {
    public static void train(NeuralNetwork network, List<TrainingInput> samples, double h, double learningRate) {
        List<NeuralLayer> layers = new ArrayList<>(network.layers);
        layers.removeFirst();
        for (NeuralLayer neuralLayer : layers) {
            double originalCost = network.cost(samples);
            learn(neuralLayer, network, samples, h, originalCost, learningRate);
        }
    }

    public static void learn(NeuralLayer layer, NeuralNetwork network, List<TrainingInput> samples, double h, double currentCost, double learningRate) {
        for (Neuron neuronIn : layer.before.neurons) {
            for (Neuron neuronOut : layer.neurons) {
                double value = neuronIn.connectionWeights.get(neuronOut).doubleValue();
                neuronIn.connectionWeights.put(neuronOut, BigDecimal.valueOf(value + h));

                double cost = network.cost(samples);
                double delta = cost - currentCost;

                if (delta > 0) {
                    //Wrong direction
                    value -= learningRate;
                } else {
                    //Correct direction
                    value += learningRate;
                }

                neuronIn.connectionWeights.put(neuronOut, BigDecimal.valueOf(value));

                currentCost = network.cost(samples);
            }
        }

        currentCost =  network.cost(samples);

        for (Neuron neuron : layer.neurons) {
            neuron.bias += h;
            double currentCost0 = network.cost(samples);
            double delta = (currentCost0 - currentCost);
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