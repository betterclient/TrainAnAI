package io.github.betterclient.ai.model;

import io.github.betterclient.ai.neural.NeuralNetwork;
import io.github.betterclient.ai.object.NeuralNetworkTrainer;
import io.github.betterclient.ai.training.TrainingInput;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class Model {
    public int epochs, trainingSampleSize;
    public float h, learningRate;
    public final String name, description;
    public int[] layerSizes;
    public NeuralNetwork network;

    public Model(String name, String description, int epochs, float h, float learningRate, int trainingSampleSize, int[] layerSizes) {
        this.epochs = epochs;
        this.h = h;
        this.learningRate = learningRate;
        this.name = name;
        this.description = description;
        this.trainingSampleSize = trainingSampleSize;
        this.layerSizes = layerSizes;
    }


    public abstract void updateData();
    public abstract String getInputForData(String data);
    public abstract List<TrainingInput> getTrainingSamples();

    public final void train() {
        List<Integer> sizes = new ArrayList<>();
        Arrays.stream(layerSizes).forEach(sizes::add);
        sizes.removeFirst(); sizes.removeLast();

        network = new NeuralNetworkTrainer(
                layerSizes[0],
                sizes.stream().mapToInt(x -> x).toArray(),
                layerSizes[layerSizes.length - 1],

                this.getTrainingSamples(),

                this.epochs,
                this.h,
                this.learningRate,
                true //Change this
        ).train();
    }
}