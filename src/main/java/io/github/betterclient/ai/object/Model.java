package io.github.betterclient.ai.object;

import io.github.betterclient.ai.neural.NeuralNetwork;
import io.github.betterclient.ai.training.TrainingInput;
import org.teavm.jso.dom.html.HTMLDocument;
import org.teavm.jso.dom.html.HTMLInputElement;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class Model {
    public int epochs, trainingSampleSize;
    public float learningRate;
    public final String name, description;
    public int[] layerSizes;
    public NeuralNetwork network;

    public Model(String name, String description, int epochs, float learningRate, int trainingSampleSize, int[] layerSizes) {
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.name = name;
        this.description = description;
        this.trainingSampleSize = trainingSampleSize;
        this.layerSizes = layerSizes;
    }

    public abstract void updateData();
    public abstract String getInputForData(String data);
    public abstract String getOutput();
    public abstract List<TrainingInput> getTrainingSamples();

    public final void train() {
        List<Integer> sizes = new ArrayList<>();
        Arrays.stream(layerSizes).forEach(sizes::add);
        sizes.removeFirst(); sizes.removeLast();

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(
                layerSizes[0],
                sizes.stream().mapToInt(x -> x).toArray(),
                layerSizes[layerSizes.length - 1],

                this.getTrainingSamples(),

                this.epochs,
                this.learningRate,
                true //Change this
        );
        trainer.printNetworkSize();
        network = trainer.train();
    }

    protected final int getSlider(String sliderID) {
        HTMLInputElement inputElement = (HTMLInputElement) HTMLDocument.current().getElementById(sliderID);
        return Integer.parseInt(inputElement.getValue());
    }
}