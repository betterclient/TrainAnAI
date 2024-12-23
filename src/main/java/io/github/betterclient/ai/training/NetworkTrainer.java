package io.github.betterclient.ai.training;

import io.github.betterclient.ai.neural.NeuralNetwork;

import java.util.List;

public class NetworkTrainer {
    /**
     * Train the given network using the given training data
     * <p></p>
     * TODO: <a href="https://youtu.be/hfMk-kjRv4c?t=938">Continue watching the explanation</a>
     * @param network the network to train (weights and biases)
     * @param data Training Data
     */
    public static void train(NeuralNetwork network, List<TrainingInput> data, int epochs, double h, double learnRate) {
        for (int i = 0; i < epochs; i++) {
            if(network.learn(data, h, learnRate) == 0) {
                System.out.println("Training successful in " + i + " tries");
                break;
            }

            if (i % (epochs / 10) == 0) {
                //*System.out.println("Training " + ((i / (double)epochs) * 100) + "% complete.");
            }
        }

        //System.out.println("Training 100.0% complete.");
    }

    private static double getCost(NeuralNetwork network, TrainingInput data) {
        double[] actualOutput = network.forward(data.inputs);
        double cost = 0;

        for (int i = 0; i < actualOutput.length; i++) {
            double error = actualOutput[i] - data.expected[i];
            cost += error * error;
        }

        return cost;
    }

    public static double getCost(NeuralNetwork network, List<TrainingInput> inputs) {
        double cost = 0;
        for (TrainingInput input : inputs) {
            cost += getCost(network, input);
        }
        return cost;
    }
}
