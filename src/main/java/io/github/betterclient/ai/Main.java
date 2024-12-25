package io.github.betterclient.ai;

import io.github.betterclient.ai.object.NeuralNetworkTrainer;
import io.github.betterclient.ai.training.NetworkTrainer;
import io.github.betterclient.ai.neural.NeuralNetwork;
import io.github.betterclient.ai.training.TrainingInput;

import java.text.DecimalFormat;
import java.util.*;
import java.util.function.Function;

public class Main {
    static Map<Float, Float> SIGMOID_CACHE = new HashMap<>();
    public static final Function<Float, Float> ACTIVATION_FUNCTION = x -> SIGMOID_CACHE.computeIfAbsent(x, Main::sigmoid);

    private static float sigmoid(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }

    public static void main(String[] args) {
        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(
                1, new int[] {40}, 2,
                createTrainingData(),
                100,
                0.01f,
                0.01f,
                true
        );
        NeuralNetwork network = trainer.train();
        System.out.println("Cost of network: " + trainer.getCost(network) + " (closer to 0 is better)");
        trainer.printNetworkSize();

        System.out.println("Input \"50\": " + display(network.forward(new float[]{50})) + " expected -> [1.0, 0.0]");
        System.out.println("Input \"1\": " + display(network.forward(new float[]{1})) + " expected -> [1.0, 0.0]");
        System.out.println("Input \"200\": " + display(network.forward(new float[]{200})) + " expected -> [0.0, 1.0]");
        System.out.println("Input \"100\": " + display(network.forward(new float[]{100})) + " expected -> [0.0, 1.0]");

        System.out.println("\nCases that aren't in training data: ");
        System.out.println("Input \"300\": " + display(network.forward(new float[]{300})) + " expected -> [0.0, 1.0]");
        System.out.println("Input \"-1\": " + display(network.forward(new float[]{-1})) + " expected -> [1.0, 0.0]");

        System.out.println("\nExtreme cases that aren't in training: ");
        System.out.println("Input \"-1238128\": " + display(network.forward(new float[]{-1238128})) + " expected -> [1.0, 0.0]");
        System.out.println("Input \"12932209\": " + display(network.forward(new float[]{12932209})) + " expected -> [0.0, 1.0]");
    }

    private static String display(float[] forward) {
        DecimalFormat format = new DecimalFormat("0.00");
        return "[" + format.format(forward[0]) + ", " + format.format(forward[1]) + "]";
    }

    private static List<TrainingInput> createTrainingData() {
        List<TrainingInput> data = new ArrayList<>();

        for (int i = -500; i < 500; i++) {
            data.add(new TrainingInput(
                    new float[] {i},
                    new float[] {i <= 100 ? 1 : 0, i > 100 ? 1 : 0}
            ));
        }

        return data;
    }
}
