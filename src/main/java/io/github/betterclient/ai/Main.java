package io.github.betterclient.ai;

import io.github.betterclient.ai.model.SmallLanguageModel;
import io.github.betterclient.ai.object.Model;

public class Main {
    public static ActivationFunction ACTIVATION_FUNCTION = ActivationFunction.GELU;

    public static void main(String[] args) {
        Model model = new SmallLanguageModel();
        model.updateData(); //simulate user inputs for training information like epochs and layer data
        model.train();

        System.out.println("Model output for \"<start> python pickle\": " + model.getInputForData("<start> python pickle"));
    }
}