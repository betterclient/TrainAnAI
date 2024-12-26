package io.github.betterclient.ai;

import io.github.betterclient.ai.model.IsBiggerThan100Model;
import io.github.betterclient.ai.model.Model;

import java.util.*;
import java.util.function.Function;

public class Main {
    static Map<Float, Float> SIGMOID_CACHE = new HashMap<>();
    public static final Function<Float, Float> ACTIVATION_FUNCTION = x -> SIGMOID_CACHE.computeIfAbsent(x, Main::sigmoid);

    private static float sigmoid(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }

    public static void main(String[] args) {
        Model model = new IsBiggerThan100Model();
        model.updateData(); //simulate user inputs for training information like epochs and layer data
        model.train();

        Scanner scanner = new Scanner(System.in);
        String out;
        System.out.print("Please input a float: ");
        while(!(out = scanner.nextLine()).equals("exit")) {
            System.out.println(model.getInputForData(out));
            System.out.print("Please input a float: ");
        }
    }
}
