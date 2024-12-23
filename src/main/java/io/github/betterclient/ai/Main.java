package io.github.betterclient.ai;

import io.github.betterclient.ai.training.NetworkTrainer;
import io.github.betterclient.ai.neural.NeuralNetwork;
import io.github.betterclient.ai.training.TrainingInput;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class Main {
    static Map<Double, Double> SIGMOID_CACHE = new HashMap<>();
    public static final Function<Double, Double> ACTIVATION_FUNCTION = x -> SIGMOID_CACHE.computeIfAbsent(x, Main::sigmoid);
    private static double sigmoid(double x) {
        return (1 / (1 + Math.exp(-x)));
    }

    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork(new int[] {1, 1});

        //Train the network for results
        NetworkTrainer.train
                (
                        network,
                        createTrainingData(),
                        //Do training 100 times (bigger is more precise)
                        10,
                        //Learn slope function amount thing I don't really know (smaller is better)
                        0.1,
                        //Learning rate, configure the gradient descent's (value changes from application)
                        0.1
                );

        System.out.println(network.forward(new double[] {54})[0] + " should be close to 0");
        System.out.println(network.forward(new double[] {101})[0] + " should be close to 1");
    }

    private static List<TrainingInput> createTrainingData() {
        List<TrainingInput> data = new ArrayList<>();

        for (int i = 0; i < 200; i++) {
            data.add(new TrainingInput(
                    new double[] {i},
                    new double[] {i > 100 ? 1 : 0}
            ));
        }

        return data;
    }
}
