package io.github.betterclient.ai.training;

import io.github.betterclient.ai.neural.NeuralNetwork;

public class NetworkTrainer {

    /**
     * Train the given network using the input and target
     * <p></p>
     * TODO: <a href="https://youtu.be/hfMk-kjRv4c?t=938">Continue watching the explanation</a>
     * @param network the network to train (weights and biases)
     * @param inputs expected inputs
     * @param targets expected outputs for those inputs
     */
    public static void train(NeuralNetwork network, double[][] inputs, double[][] targets) {
        TrainingDataPoint point = network.convert(inputs, targets);

        System.out.println(point.cost());
    }

    private static double getCost(NeuralNetwork network, double[] input, double[] expectedOutput) {
        double[] actualOutput = network.forward(input);
        double cost = 0;

        for (int i = 0; i < actualOutput.length; i++) {
            double error = actualOutput[i] - expectedOutput[i];
            cost += error * error;
        }

        return cost;
    }

    public static double getCost(NeuralNetwork network, double[][] input, double[][] expectedOutput) {
        double cost = 0;

        for (int i = 0; i < input.length; i++) {
            cost += getCost(network, input[i], expectedOutput[i]);
        }

        return cost;
    }
}
