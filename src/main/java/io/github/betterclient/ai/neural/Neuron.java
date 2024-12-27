package io.github.betterclient.ai.neural;

import io.github.betterclient.ai.training.NetworkTrainer;
import io.github.betterclient.ai.training.TrainingInput;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Neuron {
    public Map<Neuron, Float> connectionWeights = new HashMap<>();
    public Map<Neuron, Float> connectionWeightCosts = new HashMap<>();
    public float bias;
    public float biasCost;

    public Neuron(float bias) {
        this.bias = bias;
    }

    public void initConnections(NeuralLayer layer) {
        for (Neuron neuron : layer.neurons) {
            connectionWeights.put(neuron, 0F);
        }
    }

    //---------------------------------------TRAINING-----------------------------------------
    /**
     * Neurons will turn themselves off after not having effect.
     */
    public boolean hasEffect = true;
    public int noEffectCount = 0;

    public float learn(NeuralNetwork network, List<TrainingInput> data, float h, float learningRate, Neuron neuronOut, float currentCost) {
        if (!hasEffect) {
            noEffectCount++;

            if(noEffectCount == 4) {
                noEffectCount = 0;
                hasEffect = true;
            } else {
                return currentCost;
            }
        }

        float value = this.connectionWeights.get(neuronOut);
        float delta = getDelta(network, data, neuronOut, currentCost, h, value);

        int finalI = 1;
        for (int i = 1; i < 4; i++) {
            delta = getDelta(network, data, neuronOut, currentCost, h * i, value);
            if (delta <= 0) {
                finalI = i - 1; //Return the last non negative delta
                hasEffect = false;
                break;
            }
        }

        if(delta > 0) {
            //Wrong direction
            value -= learningRate * finalI;
        } else {
            //Correct direction
            value += learningRate * finalI;
        }

        this.connectionWeights.put(neuronOut, value);

        float last = NetworkTrainer.getCost(network, data);

        if ((currentCost - last) > (h / learningRate)) {
            hasEffect = false;
        }

        return last;
    }

    @SuppressWarnings("all")
    private float getDelta(NeuralNetwork network, List<TrainingInput> data, Neuron forN, float currentCost, float h, float value) {
        this.connectionWeights.put(forN, value + h);
        float cost = NetworkTrainer.getCost(network, data);
        this.connectionWeights.put(forN, value);

        return cost - currentCost;
    }
}