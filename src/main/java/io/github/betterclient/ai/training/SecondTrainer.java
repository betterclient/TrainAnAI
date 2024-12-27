package io.github.betterclient.ai.training;

import io.github.betterclient.ai.neural.NeuralLayer;
import io.github.betterclient.ai.neural.NeuralNetwork;
import io.github.betterclient.ai.neural.Neuron;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class SecondTrainer {
    public static void train(NeuralNetwork network, List<TrainingInput> samples, float learnRate) {
        float startingCost = network.cost(samples);
        System.out.println(startingCost + " starting");

        for (NeuralLayer layer : network.layersW()) {
            for (Neuron neuronIn : layer.before.neurons) {
                for (Neuron neuronOut : layer.neurons) {
                    Map<Neuron, Float> weights = neuronIn.connectionWeights;

                    neuronIn.connectionWeightCosts.put(neuronOut,
                            slope(x -> {
                                float old = weights.get(neuronOut);
                                weights.put(neuronOut, x);
                                float newCost = network.cost(samples);
                                weights.put(neuronOut, old);
                                return newCost;
                            }, weights.get(neuronOut))
                    );
                }
            }

            for (Neuron neuron : layer.neurons) {
                neuron.biasCost = slope(x -> {
                    float old = neuron.bias;
                    neuron.bias = x;
                    float newCost = network.cost(samples);
                    neuron.bias = old;
                    return newCost;
                }, neuron.bias);
            }
        }

        for (NeuralLayer layer : network.layersW()) {
            layer.applyGradients(learnRate);
        }

        System.out.println(network.cost(samples) + " ending");
    }

    private static float slope(Function<Float, Float> function, float val) {
        float h = 0.01f;
        //calculate the slope using f(x + h) - f(x)
        float delta = function.apply(val + h) - function.apply(val);
        return delta / h;
    }
}