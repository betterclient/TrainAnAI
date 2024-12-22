package io.github.betterclient.ai.neural;

import io.github.betterclient.ai.Main;

import java.util.ArrayList;
import java.util.List;

public class NeuralLayer {
    //LinkedList structure
    public NeuralLayer before;
    public NeuralLayer after;
    public List<Neuron> neurons = new ArrayList<>();

    public NeuralLayer(int layerCount) {
        for (int i = 0; i < layerCount; i++) {
            neurons.add(new Neuron(Math.random() - .5));
        }
    }

    public double[] forward(double[] inputs) {
        double[] weightedInputs = new double[neurons.size()];

        for (int i = 0; i < neurons.size(); i++) {
            double weightedInput = neurons.get(i).bias;

            for (int i1 = 0; i1 < before.neurons.size(); i1++) {
                weightedInput += inputs[i1] * before.neurons.get(i1).connectionWeights.get(neurons.get(i));
            }
            weightedInputs[i] = Main.ACTIVATION_FUNCTION.apply(weightedInput);
        }

        return weightedInputs;
    }
}
