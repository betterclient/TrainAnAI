package io.github.betterclient.ai.neural;

import io.github.betterclient.ai.Main;
import io.github.betterclient.ai.training.NetworkTrainer;
import io.github.betterclient.ai.training.TrainingInput;

import java.util.ArrayList;
import java.util.List;

public class NeuralLayer {
    //LinkedList structure
    public NeuralLayer before;
    public NeuralLayer after;
    public List<Neuron> neurons = new ArrayList<>();

    public NeuralLayer(int layerCount) {
        for (int i = 0; i < layerCount; i++) {
            neurons.add(new Neuron(0));
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

    public void applyGradients(double learnRate) {
        for (Neuron neuronOut : this.neurons) {
            neuronOut.bias -= neuronOut.costGradientB * learnRate;

            for (Neuron neuronIn : this.before.neurons) {
                neuronIn.connectionWeights.compute(
                        neuronOut,
                        (k, old) -> {
                            System.out.println(old + " to -> " + (old - (neuronIn.costGradientW.get(neuronOut) * learnRate)) + " (did - " + (neuronIn.costGradientW.get(neuronOut) * learnRate) + ")");
                            return old - (neuronIn.costGradientW.get(neuronOut) * learnRate);
                        }
                );
            }
        }
    }

    public void learn(NeuralNetwork network, List<TrainingInput> data, double h, double originalCost) {
        for (Neuron neuronIn : this.before.neurons) {
            for (Neuron neuronOut : this.neurons) {
                double value = neuronIn.connectionWeights.get(neuronOut);
                neuronIn.connectionWeights.put(neuronOut, value + h);

                //Put change in costGradientW
                System.out.println("Original: " + originalCost + " new: " + NetworkTrainer.getCost(network, data) + " h: " + h + " out");
                neuronIn.costGradientW.put(neuronOut, (NetworkTrainer.getCost(network, data) - originalCost) / h);

                neuronIn.connectionWeights.put(neuronOut, value);
            }
        }

        for (Neuron neuron : this.neurons) {
            neuron.bias += h;
            neuron.costGradientB = (NetworkTrainer.getCost(network, data) - originalCost) / h;
            neuron.bias -= h;
        }
    }
}
