package io.github.betterclient.ai.neural;

import io.github.betterclient.ai.Main;
import io.github.betterclient.ai.training.PreCalculation;

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

    public float[] forward(float[] inputs) {
        float[] weightedInputs = new float[neurons.size()];

        for (int i = 0; i < neurons.size(); i++) {
            float weightedInput = neurons.get(i).bias;

            for (int i1 = 0; i1 < before.neurons.size(); i1++) {
                weightedInput += inputs[i1] * before.neurons.get(i1).connectionWeights.get(neurons.get(i));
            }
            weightedInputs[i] = Main.ACTIVATION_FUNCTION.func.apply(weightedInput);
        }

        return weightedInputs;
    }

    public void applyGradients(float learnRate) {
        for (Neuron neuron : neurons) {
            neuron.bias -= neuron.biasCost * learnRate;
            for (Neuron neuronIn : before.neurons) {
                neuronIn.connectionWeights.put(
                        neuron,
                        neuronIn.connectionWeights.get(neuron) - (neuronIn.connectionWeightCosts.get(neuron) * learnRate)
                );
            }
        }
    }

    public float[] calculate(PreCalculation calculation, float[] expected) {
       float[] outNodes = new float[expected.length];

        for (int i = 0; i < outNodes.length; i++) {
            float costDerivative = nodeCostDerivative(calculation.activations[i], expected[i]);
            float activationDerivative = Main.ACTIVATION_FUNCTION.derivative.apply(calculation.weightedInputs[i]);
            outNodes[i] = activationDerivative * costDerivative;
        }

        return outNodes;
    }

    private float nodeCostDerivative(float found, float expected) {
        return 2 * (found - expected);
    }
}