package io.github.betterclient.ai.training;

import io.github.betterclient.ai.neural.NeuralNetwork;

import java.util.List;

public class NetworkTrainer {
    private static float lastCost;

    /**
     * Train the given network using the given training data
     * @param network the network to train (weights and biases)
     * @param data Training Data
     */
    public static void train(NeuralNetwork network, List<TrainingInput> data, int epochs, float h, float learnRate) {
        long start = System.currentTimeMillis();
        for (int i = 0; i < epochs; i++) {
            if(network.learn(data, h, learnRate) == 0) {
                //That case where it somehow got to a perfect accuracy
                printTime(start);
                System.out.println("Training successful in " + i + " tries");
                break;
            }

            System.out.println("Epoch: " + i + "/" + epochs + " complete. Current Cost is " + lastCost);

            if (i % 20 == 0) {
                System.gc(); //Just to clear memory a little
            }
        }

        printTime(start);
    }

    private static void printTime(long start) {
        System.out.println("NeuralNetwork.learn completed in " + (System.currentTimeMillis() - start) / 1000F + "seconds.");
    }

    private static float getCost(NeuralNetwork network, TrainingInput data) {
        float[] actualOutput = network.forward(data.inputs);
        float cost = 0;

        for (int i = 0; i < actualOutput.length; i++) {
            float error = actualOutput[i] - data.expected[i];
            cost += error * error;
        }

        return cost;
    }

    public static float getCost(NeuralNetwork network, List<TrainingInput> inputs) {
        float cost = 0;
        for (TrainingInput input : inputs) {
            cost += getCost(network, input);
        }
        lastCost = cost;
        return cost;
    }
}
