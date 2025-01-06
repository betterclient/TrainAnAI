package io.github.betterclient.ai.object;

import io.github.betterclient.ai.neural.NeuralNetwork;
import io.github.betterclient.ai.training.NetworkTrainer;
import io.github.betterclient.ai.training.TrainingInput;
import io.github.betterclient.ai.web.TrainingStatus;
import org.teavm.jso.browser.Window;
import org.teavm.jso.dom.html.HTMLDocument;
import org.teavm.jso.dom.html.HTMLElement;

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
public record NeuralNetworkTrainer(int inputLength, int[] hiddenLayerLengths, int outputLength, List<TrainingInput> trainingSamples, int epochs, double learningRate,
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

        remainingEpochs = epochs;
        train(network);

        return network;
    }

    private void train(NeuralNetwork network) {
        double ending = NetworkTrainer.train(network, trainingSamples, learningRate);
        if (displayTraining) {
            System.out.println("Epoch: " + (epochs - remainingEpochs) + " complete!");
        }
        String s = (epochs - remainingEpochs) + "/" + epochs;

        HTMLDocument document = HTMLDocument.current();
        HTMLElement trainingStatus = document.getElementById("TRAINING_STATUS");
        trainingStatus.setInnerText("Training (website may lag): " + s);

        if (remainingEpochs-- > 0) Window.requestAnimationFrame(timestamp -> train(network));
        else {
            if (TrainingStatus.TRAINED_MODEL != null) {
                trainingStatus.setInnerText("Trained");
                trainingStatus.getStyle().setProperty("color", "green");
            } else {
                TrainingStatus.untrain();
            }
        }
    }

    public static int remainingEpochs;

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
