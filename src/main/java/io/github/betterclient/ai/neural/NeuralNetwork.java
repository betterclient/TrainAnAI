package io.github.betterclient.ai.neural;

import io.github.betterclient.ai.training.NetworkTrainer;
import io.github.betterclient.ai.training.TrainingInput;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NeuralNetwork {
    public List<NeuralLayer> layers = new ArrayList<>();

    public NeuralNetwork(int[] layerCounts) {
        for (int layerCount : layerCounts) {
            layers.add(new NeuralLayer(layerCount));
        }

        NeuralLayer before = null;
        int index = 0;
        for (NeuralLayer layer : layers) {
            index++;

            layer.before = before;
            before = layer;

            if (layer != layers.getLast()) {
                layer.after = layers.get(index);
            }
        }

        for (NeuralLayer layer : layers) {
            if (layer != layers.getLast()) {
                for (Neuron neuron : layer.neurons) {
                    neuron.initConnections(layers.get(layers.indexOf(layer) + 1));
                }
            }
        }
    }

    public double learn(List<TrainingInput> data, float h, float learnRate) {
        List<NeuralLayer> layers = new ArrayList<>(this.layers);
        layers.removeFirst();
        for (NeuralLayer neuralLayer : layers) {
            float originalCost = NetworkTrainer.getCost(this, data);
            neuralLayer.learn(this, data, h, originalCost, learnRate);
        }

        return NetworkTrainer.getCost(this, data);
    }

    public float[] forward(float[] inputs) {
        for (int i = 1; i < layers.size(); i++) {
            NeuralLayer layer = layers.get(i);

            var out = preCalculatedNeuralLayers.get(layer);
            if (out != null) {
                //-------pre calculated . get( precalculated index ------object ---------- for inputs )
                float[] neww = out.get(inputs);

                if (neww != null) {
                    inputs = neww; //This layer has pre-calculated forwarding for this input
                    //yay!
                    continue;
                }
            }
            inputs = layer.forward(inputs);
        }
        return inputs;
    }

    //Give the index of the precalculated call
    public Map<NeuralLayer, Map<float[], float[]>> preCalculatedNeuralLayers = new HashMap<>();

    /**
     * Optimization: Figure out the output of everything before and after a specific layer
     * @param layerToIgnore Layer to not optimize (the layer that's going to receive changes)
     * @param forData Data to optimize for
     */
    public void setupForwardingOptimization(NeuralLayer layerToIgnore, List<TrainingInput> forData) {
        for (NeuralLayer layer : this.layers) {
            if (layer == layerToIgnore) continue; //Ignore, lol
            preCalculatedNeuralLayers.put(layer, new HashMap<>());
        }

        float[][] currentInputsForAllData = new float[forData.size()][];
        for (int i = 0; i < forData.size(); i++) {
            currentInputsForAllData[i] = forData.get(i).inputs; //Initialize inputs
        }

        for (int index = 0; index < currentInputsForAllData.length; index++) {
            for (int i = 1; i < this.layers.size(); i++) {
                NeuralLayer layer = layers.get(i);
                float[] forward = layer.forward(currentInputsForAllData[index]);

                if (preCalculatedNeuralLayers.containsKey(layer)) {

                    //precalculate inputs!!
                    preCalculatedNeuralLayers.get(layer)
                            .put(currentInputsForAllData[index], forward);

                    currentInputsForAllData[index] = forward;
                    continue;
                }
                currentInputsForAllData[index] = forward;
            }
        }
    }

    public void stopOptimizations() {
        preCalculatedNeuralLayers.clear();
    }
}