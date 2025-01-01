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

    public void updateGradients(PreCalculation calculation, float[] nodeValues) {
        for (int nodeOut = 0; nodeOut < neurons.size(); nodeOut++) {
            Neuron neuronOut = neurons.get(nodeOut);
            for (int nodeIn = 0; nodeIn < before.neurons.size(); nodeIn++) {
                Neuron neuronIn = before.neurons.get(nodeIn);

                Float old = neuronIn.connectionWeightCosts.get(neuronOut);
                neuronIn.connectionWeightCosts.put(
                        neuronOut,

                        (old == null ? 0 : old)
                        +
                        calculation.inputs[nodeIn] * nodeValues[nodeOut]
                );
            }

            neurons.get(nodeOut).biasCost += nodeValues[nodeOut];
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

    public float[] preCalculate(float[] inputs, PreCalculation calculation) {
        calculation.inputs = inputs;

        int index = 0;
        for (Neuron neuronOut : neurons) {
            float weighted = neuronOut.bias;

            for (int nodeIn = 0; nodeIn < before.neurons.size(); nodeIn++) {
                Neuron neuronIn = before.neurons.get(nodeIn);

                weighted += inputs[nodeIn] * neuronIn.connectionWeights.get(neuronOut);
            }

            calculation.weightedInputs[index] = weighted;
            index++;
        }

        //Activate
        for (int i = 0; i < calculation.activations.length; i++) {
            calculation.activations[i] = Main.ACTIVATION_FUNCTION.func.apply(calculation.weightedInputs[i]);
        }

        return calculation.activations;
    }

    public float[] calculateHiddenLayerNodeValues(PreCalculation calculation, NeuralLayer oldLayer, float[] oldNodeValues) {
        float[] newNodeValues = new float[neurons.size()];

        for (int newNodeIndex = 0; newNodeIndex < newNodeValues.length; newNodeIndex++) {
            float newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.length; oldNodeIndex++) {
                float weightedInputDerivative =
                        oldLayer.before.neurons.get(newNodeIndex)
                                .connectionWeights.get(
                                        oldLayer.neurons.get(oldNodeIndex)
                                );

                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }
            newNodeValue *= Main.ACTIVATION_FUNCTION.derivative.apply(calculation.weightedInputs[newNodeIndex]);
            newNodeValues[newNodeIndex] = newNodeValue;
        }

        return newNodeValues;
    }

    public void clearGradients() {
        for (Neuron neuron : neurons) {
            neuron.biasCost = 0;
            neuron.connectionWeightCosts.clear();
        }
    }
}