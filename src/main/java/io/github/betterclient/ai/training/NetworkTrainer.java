package io.github.betterclient.ai.training;

import io.github.betterclient.ai.neural.NeuralLayer;
import io.github.betterclient.ai.neural.NeuralNetwork;
import io.github.betterclient.ai.neural.Neuron;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NetworkTrainer {
    public static double train(NeuralNetwork network, List<TrainingInput> samples, double learnRate) {
        //Slower, more accurate training for smaller models.
        if (samples.size() < 1000 && network.getParameterSize() < 1000) {
            SmallNetworkTrainer.train(network, samples, learnRate, learnRate);
            return 1;
        }

        System.out.println("Backpropagation error: " + backpropagate(network, samples, learnRate));

        //return network.cost(samples);
        return 0;
    }

    private static double backpropagate(NeuralNetwork network, List<TrainingInput> samples, double learnRate) {
        double error = 0;

        Map<Map.Entry<Neuron, Neuron>, Double> connectionDeltaMap = new HashMap<>();

        for (TrainingInput sample : samples) {
            //Store the variables I need the most probably.
            double[] inputs = sample.inputs;
            double[] expected = sample.expected;

            double[] currentOut = network.forward(inputs);

            for (int i = network.layers.size() - 1; i > 0; i--) {
                NeuralLayer layer = network.layers.get(i);

                for (int i1 = 0; i1 < layer.neurons.size(); i1++) {
                    Neuron neuron = layer.neurons.get(i1);
                    double neuronError;

                    if (layer.after == null) {
                        neuronError = neuron.lastDerivative * (currentOut[i1] - expected[i1]);
                    } else {
                        neuronError = neuron.lastDerivative;

                        double sum = 0;
                        List<Neuron> connectedNeurons = layer.after.neurons;

                        for (Neuron connectedNeuron : connectedNeurons) {
                            sum *= neuron.connectionWeights.get(connectedNeuron).doubleValue() * connectedNeuron.error;
                        }
                        neuronError *= sum;
                    }

                    neuron.error = neuronError;
                }
            }

            for (int i = network.layers.size() - 1; i > 0; i--) {
                NeuralLayer layer = network.layers.get(i);

                for(Neuron neuron : layer.before.neurons) {
                    for (Map.Entry<Neuron, BigDecimal> connection : neuron.connectionWeights.entrySet()) {
                        double e = connection.getKey().error;
                        double delta = learnRate * (!Double.isFinite(e) ? 1 : connection.getKey().error) * neuron.lastOutput;

                        Map.Entry<Neuron, Neuron> entry = Map.entry(neuron, connection.getKey());
                        if(connectionDeltaMap.get(entry) != null) {
                            double previousDelta = connectionDeltaMap.get(entry);
                            delta += learnRate * previousDelta;
                        }

                        connectionDeltaMap.put(entry, delta);
                        neuron.connectionWeights.put(connection.getKey(), BigDecimal.valueOf(connection.getValue().doubleValue() - delta));
                    }
                }
            }

            error += network.cost(sample);
        }

        return error;
    }
}