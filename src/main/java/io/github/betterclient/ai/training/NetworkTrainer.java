package io.github.betterclient.ai.training;

import io.github.betterclient.ai.neural.NeuralLayer;
import io.github.betterclient.ai.neural.NeuralNetwork;

import java.util.List;

public class NetworkTrainer {
    private static float lastCost;

    /**
     * Train the given network using the given training data
     *
     * @param network         the network to train (weights and biases)
     * @param data            Training Data
     * @param displayTraining
     */
    public static void train(NeuralNetwork network, List<TrainingInput> data, int epochs, float h, float learnRate, boolean displayTraining) {
        long start = System.currentTimeMillis();

        float cost = getCost(network, data);

        for (int i = 1; i < network.layers.size(); i++) {
            NeuralLayer layer = network.layers.get(i);

            network.setupForwardingOptimization(layer.before, data); //setup optimizing the layer before

            for (int i1 = 0; i1 < epochs; i1++) {
                layer.learn(network, data, h, cost, learnRate);

                if ((cost - lastCost) < h) {
                    //Too less of a change to actually care about
                    break;
                }

                if (displayTraining) System.out.println("Epoch: " + i1 + "/" + epochs + " complete. Current Cost is " + lastCost + " Delta: " + (cost - lastCost));

                if (i1 % 20 == 0) {
                    System.gc(); //Just to clear memory a little
                }

                cost = lastCost;
            }

            network.stopOptimizations();
        }

        printTime(start);
        if(true) return;

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
