package io.github.betterclient.ai.neural;

import io.github.betterclient.ai.training.PreCalculation;
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

    public float[] forward(float[] inputs) {
        for (int i = 1; i < layers.size(); i++) {
            NeuralLayer layer = layers.get(i);

            inputs = layer.forward(inputs);
        }
        return inputs;
    }

    public float cost(List<TrainingInput> samples) {
        float cost = 0;
        for (TrainingInput data : samples) {
            float[] actualOutput = this.forward(data.inputs);

            for (int i = 0; i < actualOutput.length; i++) {
                float error = actualOutput[i] - data.expected[i];
                cost += error * error;
            }
        }
        return cost;
    }

    public List<NeuralLayer> layersW() {
        List<NeuralLayer> a = new ArrayList<>(layers);
        a.removeFirst();
        return a;
    }

    public void updateAllGradients(TrainingInput sample) {
        calculateOutputs(sample.inputs);

        NeuralLayer outputLayer = this.layers.getLast();
        PreCalculation calculation = preCalculations.get(outputLayer);

        //Backpropagation
        float[] values = outputLayer.calculate(calculation, sample.expected);
        outputLayer.updateGradients(calculation, values);

        for (int i = layers.size() - 2; i >= 1; i--) {
            NeuralLayer hiddenLayer = layers.get(i);
            calculation = preCalculations.get(hiddenLayer);

            values = hiddenLayer.calculateHiddenLayerNodeValues(calculation, outputLayer, values);
            hiddenLayer.updateGradients(calculation, values);
        }

        preCalculations.clear();
    }

    private final Map<NeuralLayer, PreCalculation> preCalculations = new HashMap<>();
    private void calculateOutputs(float[] inputs) {
        for (int i = 1; i < layers.size(); i++) {
            NeuralLayer layer = layers.get(i);

            PreCalculation calculation = new PreCalculation(layer);
            inputs = layer.preCalculate(inputs, calculation);
            preCalculations.put(layer, calculation);
        }
    }
}