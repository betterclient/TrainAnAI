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
            neurons.add(new Neuron(0f));
        }
    }

    public float[] forward(float[] inputs) {
        float[] weightedInputs = new float[neurons.size()];

        for (int i = 0; i < neurons.size(); i++) {
            float weightedInput = neurons.get(i).bias;

            for (int i1 = 0; i1 < before.neurons.size(); i1++) {
                weightedInput += inputs[i1] * before.neurons.get(i1).connectionWeights.get(neurons.get(i));
            }
            weightedInputs[i] = Main.ACTIVATION_FUNCTION.apply(weightedInput);
        }

        return weightedInputs;
    }

    public void learn(NeuralNetwork network, List<TrainingInput> data, float h, float currentCost, float learningRate) {
        for (Neuron neuronIn : this.before.neurons) {
            for (Neuron neuronOut : this.neurons) {
                currentCost = neuronIn.learn(network, data, h, learningRate, neuronOut, currentCost);
            }
        }

        currentCost = NetworkTrainer.getCost(network, data);

        for (Neuron neuron : this.neurons) {
            neuron.bias += h;
            float currentCost0 = NetworkTrainer.getCost(network, data);
            float delta = (currentCost0 - currentCost);
            if(delta != 0) {
                //Change happened

                if(delta > 0) {
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