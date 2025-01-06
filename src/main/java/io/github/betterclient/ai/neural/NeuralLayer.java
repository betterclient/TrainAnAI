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
            neurons.add(new Neuron((float) (Math.random() - 0.5f)));
        }
    }

    public double[] forward(double[] inputs) {
        double[] weightedInputs = new double[neurons.size()];

        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            double weightedInput = neuron.bias;

            for (int i1 = 0; i1 < before.neurons.size(); i1++) {
                weightedInput += inputs[i1] * before.neurons.get(i1).connectionWeights.get(neuron).doubleValue();
            }
            neuron.lastDerivative = Main.ACTIVATION_FUNCTION.derivative.apply(weightedInput);
            weightedInputs[i] = Main.ACTIVATION_FUNCTION.func.apply(weightedInput);
            neuron.lastOutput = weightedInputs[i];
        }

        return weightedInputs;
    }
}