package io.github.betterclient.ai.model;

import io.github.betterclient.ai.neural.NeuralNetwork;
import io.github.betterclient.ai.training.TrainingInput;

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
                0.01f,
                200, //Minimum sample size is 200 for this model
                new int[] {1, 20, 2}
        );
    }

    @Override
    public void updateData() {
        this.epochs = 100;
        this.h = 0.01f;
        this.learningRate = 0.01f;
        this.trainingSampleSize = 200;
    }

    @Override
    public String getInputForData(String data) {
        float[] out = this.network.forward(new float[] {Float.parseFloat(data)});

        boolean modelGuess = out[1] > out[0];

        return "Model guess was " + display(out) + ", it was " + ((Float.parseFloat(data) > 100 == modelGuess) ? "correct" : "not correct");
    }

    private static String display(float[] forward) {
        DecimalFormat format = new DecimalFormat("0.00");
        return "[" + format.format(forward[0]) + ", " + format.format(forward[1]) + "]";
    }

    @Override
    public List<TrainingInput> getTrainingSamples() {
        List<TrainingInput> data = new ArrayList<>();

        for (int i = -((trainingSampleSize / 2) + 100); i < ((trainingSampleSize / 2) + 100); i++) {
            data.add(new TrainingInput(
                    new float[] {i},
                    new float[] {i <= 100 ? 1 : 0, i > 100 ? 1 : 0}
            ));
        }

        return data;
    }
}