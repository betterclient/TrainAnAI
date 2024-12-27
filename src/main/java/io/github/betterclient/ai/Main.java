package io.github.betterclient.ai;

import io.github.betterclient.ai.model.SmallLanguageModel;
import io.github.betterclient.ai.object.Model;

import java.util.*;
import java.util.function.Function;

public class Main {
    static Map<Float, Float> SIGMOID_CACHE = new HashMap<>();
    public static final Function<Float, Float> ACTIVATION_FUNCTION = x -> SIGMOID_CACHE.computeIfAbsent(x, Main::sigmoid);

    static Map<Float, Float> SIGMOID_DERIVATIVE_CACHE = new HashMap<>();
    public static final Function<Float, Float> ACTIVATION_DERIVATIVE_FUNCTION = x -> SIGMOID_DERIVATIVE_CACHE.computeIfAbsent(x, Main::sigmoidDerivative);

    private static float sigmoid(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }
    private static float sigmoidDerivative(float x) {
        float sigmoid = sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }

    public static void main(String[] args) {
        Model model = new SmallLanguageModel();
        model.updateData(); //simulate user inputs for training information like epochs and layer data
        model.train();

        Scanner scanner = new Scanner(System.in);
        String out;
        System.out.print("Model input: ");
        while(!(out = scanner.nextLine()).equals("exit")) {
            System.out.println(model.getInputForData(out));
            System.out.print("Model input: ");
        }
    }
}
