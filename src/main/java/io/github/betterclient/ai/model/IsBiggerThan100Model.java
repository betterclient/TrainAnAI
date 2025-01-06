package io.github.betterclient.ai.model;

import io.github.betterclient.ai.object.Model;
import io.github.betterclient.ai.training.TrainingInput;
import org.teavm.jso.dom.html.HTMLDocument;
import org.teavm.jso.dom.html.HTMLElement;
import org.teavm.jso.dom.html.HTMLInputElement;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class IsBiggerThan100Model extends Model {
    public IsBiggerThan100Model() {
        super(
                "100>Model",
                "This is a model that tries to predict whether a given number is bigger than 100",
                100,
                0.01f,
                50,
                new int[] {1, 2, 2}
        );
    }

    public static void appendSettings(HTMLElement element) {
        element.setInnerHTML("This model doesn't have specific settings.");
    }

    public static void appendInput(HTMLElement element) {
        HTMLDocument document = HTMLDocument.current();
        HTMLInputElement input = (HTMLInputElement) document.createElement("input");
        input.setId("MODEL_INPUT_IBT100");
        input.setInnerHTML("15");
        input.setClassName("out");
        input.setType("number");
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
            if (i == 0) this.layerSizes[i] = 1;
            else if (i == this.layerSizes.length - 1) this.layerSizes[i] = 2;
            else this.layerSizes[i] = 2;
        }
    }

    @Override
    public String getInputForData(String data) {
        double[] out = this.network.forward(new double[] {Float.parseFloat(data)});

        boolean modelGuess = out[1] > out[0];

        return "Model guess was " + display(out) + ", it was " + ((Float.parseFloat(data) > 100 == modelGuess) ? "correct" : "not correct");
    }

    @Override
    public String getOutput() {
        return getInputForData(((HTMLInputElement)HTMLDocument.current().getElementById("MODEL_INPUT_IBT100")).getValue());
    }

    private static String display(double[] forward) {
        DecimalFormat format = new DecimalFormat("0.00");
        String f0 = format.format(forward[0]);
        String f1 = format.format(forward[1]);
        return "[" + (f0.equals("NaN") ? forward[0] : f0) + ", " + (f1.equals("NaN") ? forward[1] : f1) + "]";
    }

    @Override
    public List<TrainingInput> getTrainingSamples() {
        List<TrainingInput> data = new ArrayList<>();

        for (int i = 100 - trainingSampleSize; i < 100 + trainingSampleSize; i++) {
            data.add(new TrainingInput(
                    new double[] {i},
                    new double[] {i <= 100 ? 1 : 0, i > 100 ? 1 : 0}
            ));
        }

        return data;
    }
}