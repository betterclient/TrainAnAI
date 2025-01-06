package io.github.betterclient.ai.model;

import io.github.betterclient.ai.object.Model;
import io.github.betterclient.ai.training.TrainingInput;
import org.teavm.jso.dom.html.*;

import java.util.List;

public class AlgorithmPredict extends Model {
    public AlgorithmPredict() {
        super("AlgorithmPredict", "", 10, .1f, 4, new int[] {2, 2, 1});
    }

    public static void appendSettings(HTMLElement element) {
        HTMLSelectElement element1 = (HTMLSelectElement) HTMLDocument.current().createElement("select");
        element1.setId("MY_COOL_SELECTOR");
        element1.setInnerHTML("<option value=\"xor\">XOR</option>\n<option value=\"and\">AND</option>\n<option value=\"or\">OR</option>");
        element1.setClassName("TRAINING_RESET");
        element.appendChild(element1);
    }

    public static void appendInput(HTMLElement element) {
        HTMLDocument document = HTMLDocument.current();
        HTMLInputElement input = (HTMLInputElement) document.createElement("input");
        input.setId("MODEL_INPUT_APT");
        input.setPlaceholder("false, true");
        input.setClassName("out");
        input.getStyle().setProperty("height", "50px");

        element.appendChild(input);
    }

    @Override
    public void updateData() {
        int hiddenLayers = getSlider("hiddenlayers");

        this.epochs = getSlider("epochs");
        this.learningRate = getSlider("learningRate") / 10000f;
        this.trainingSampleSize = getSlider("trainingsamples");

        this.layerSizes = new int[hiddenLayers + 2];
        for (int i = 0; i < this.layerSizes.length; i++) {
            if (i == 0) this.layerSizes[i] = 2;
            else if (i == this.layerSizes.length - 1) this.layerSizes[i] = 1;
            else this.layerSizes[i] = 2;
        }
    }

    @Override
    public String getInputForData(String data) {
        String[] split = data.split(", ");
        if (split.length != 2) return "Invalid input";

        try {
            boolean v0 = Boolean.parseBoolean(split[0]);
            boolean v1 = Boolean.parseBoolean(split[1]);

            double[] input = new double[] {v0 ? 1 : 0, v1 ? 1 : 0};

            double v = this.network.forward(input)[0];
            return "Model guessed " + (v > 0.5 ? "true" : "false") + " with " + (100 - ((int)(Math.abs(v - 0.5) * 2 * 100))) + "% confidence.";
        } catch (Exception e) {
            return "Invalid input";
        }
    }

    @Override
    public String getOutput() {
        return getInputForData(((HTMLInputElement)HTMLDocument.current().getElementById("MODEL_INPUT_APT")).getValue());
    }

    @Override
    public List<TrainingInput> getTrainingSamples() {
        return switch (((HTMLSelectElement)HTMLDocument.current().getElementById("MY_COOL_SELECTOR")).getValue()) {
            case "xor" -> List.of(
                    new TrainingInput(new double[] {0, 0}, new double[] {0}),
                    new TrainingInput(new double[] {0, 1}, new double[] {1}),
                    new TrainingInput(new double[] {1, 0}, new double[] {1}),
                    new TrainingInput(new double[] {1, 1}, new double[] {0})
            );
            case "and" -> List.of(
                    new TrainingInput(new double[] {0, 0}, new double[] {0}),
                    new TrainingInput(new double[] {0, 1}, new double[] {0}),
                    new TrainingInput(new double[] {1, 0}, new double[] {0}),
                    new TrainingInput(new double[] {1, 1}, new double[] {1})
            );
            case "or" -> List.of(
                    new TrainingInput(new double[] {0, 0}, new double[] {0}),
                    new TrainingInput(new double[] {0, 1}, new double[] {1}),
                    new TrainingInput(new double[] {1, 0}, new double[] {1}),
                    new TrainingInput(new double[] {1, 1}, new double[] {1})
            );
            default -> List.of();
        };
    }
}
