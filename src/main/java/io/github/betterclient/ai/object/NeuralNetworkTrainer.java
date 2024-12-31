package io.github.betterclient.ai.object;

import io.github.betterclient.ai.neural.NeuralNetwork;
import io.github.betterclient.ai.training.NetworkTrainer;
import io.github.betterclient.ai.training.TrainingInput;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A Neural network trainer.
 * (Impact to training time & impact to precision)
 *
 * @param inputLength        the length of the input float[] array (medium)
 * @param hiddenLayerLengths the lengths of hidden layers (high & medium) (medium is better)
 * @param outputLength       the length of the output float[] array (medium)
 * @param trainingSamples    what we want the network to return for answers (extreme+ & high) (medium is better)
 * @param epochs             amount of times to refine/retrain the network (extreme & extreme) (higher is better)
 * @param learningRate       learning rate of neurons (low & medium) (lower is better)
 * @param displayTraining    whether the epoch count should be displayed
 */
public record NeuralNetworkTrainer(int inputLength, int[] hiddenLayerLengths, int outputLength, List<TrainingInput> trainingSamples, int epochs, float learningRate,
                                   boolean displayTraining) {
    /**
     * Train the network with the given information
     * @return trained network
     */
    public NeuralNetwork train() {
        List<Integer> integers = new ArrayList<>();
        integers.add(inputLength);
        Arrays.stream(hiddenLayerLengths).forEach(integers::add);
        integers.add(outputLength);

        NeuralNetwork network = new NeuralNetwork(integers.stream().mapToInt(x -> x).toArray());

        //System.out.println("TRAINING: Calculate initial cost");
        //float starting = network.cost(trainingSamples);

        for (int i = 0; i < epochs; i++) {
            float ending = NetworkTrainer.train(network, trainingSamples, learningRate);
            //System.out.println("Epoch: " + (i + 1) + "/" + epochs + " complete. Delta: " + (starting - ending) + " cost is: " + ending);
            //starting = ending; //Reuse given cost for every epoch
        }

        return network;
    }

    public void printNetworkSize() {
        List<Integer> integers = new ArrayList<>();
        integers.add(inputLength);
        Arrays.stream(hiddenLayerLengths).forEach(integers::add);
        integers.add(outputLength);
        int totalParameters = 0;
        for (int i = 0; i < integers.size() - 1; i++) {
            int currentLayerSize = integers.get(i);
            int nextLayerSize = integers.get(i + 1);
            totalParameters += currentLayerSize * nextLayerSize;
            totalParameters += nextLayerSize;
        }

        System.out.println("Running network with " + totalParameters + " parameters and " + trainingSamples.size() + " training cases");
    }
}
