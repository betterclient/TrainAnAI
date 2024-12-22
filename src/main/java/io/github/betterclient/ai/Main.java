package io.github.betterclient.ai;

import io.github.betterclient.ai.training.NetworkTrainer;
import io.github.betterclient.ai.neural.NeuralNetwork;

import java.util.function.Function;

public class Main {
    public static final Function<Double, Double> ACTIVATION_FUNCTION = x -> {
        return Math.max(0, x); //Basic RELU function
    };

    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork(new int[] {1, 1});

        //Train the network for results
        NetworkTrainer.train
                (
                        network,
                        //For input
                        createTrainingData()[0],
                        //Return input divided by 10
                        createTrainingData()[1]
                );
    }

    private static double[][][] createTrainingData() {
        double[][][] data = new double[2][200][1];

        for (int i = 0; i < 200; i++) {
            data[0][i] = new double[] {i};
            data[1][i] = new double[] {i / 10D};
        }

        return data;
    }
}
