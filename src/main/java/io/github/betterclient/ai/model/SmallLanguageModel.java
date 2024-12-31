package io.github.betterclient.ai.model;

import io.github.betterclient.ai.object.Model;
import io.github.betterclient.ai.training.TrainingInput;

import java.io.IOException;
import java.util.*;

public class SmallLanguageModel extends Model {
    public SmallLanguageModel() {
        super(
                "SLM - Small language model",
                "A really small language model that you shouldn't expect to make sense",
                1,
                0.01f,
                100, //you shouldn't be able to adjust this
                new int[] {3, 8, 5010}
        );
    }

    @Override
    public void updateData() {

    }

    @Override
    public String getInputForData(String data) {
        float[] input = new float[3];
        int index = 0;
        for (int i = data.split(" ").length - 1; i >= 0; i--) {
            if (index > 2)
                break;

            input[index] = SmallLanguageModelData.tokenize(data.split(" ")[i]);

            index++;
        }

        String out;
        StringBuilder outt = new StringBuilder();
        while (!(out = SmallLanguageModelData.untokenize(this.network.forward(input))).equals("<end>")) {
            input[0] = input[1];
            input[1] = input[2];
            input[2] = SmallLanguageModelData.tokenize(out);

            outt.append(out).append(" ");
        }
        outt.append(out).append(" ");
        return outt.toString();
    }

    @Override
    public List<TrainingInput> getTrainingSamples() {
        List<TrainingInput> data = new ArrayList<>();

        try {
            SmallLanguageModelData.parse(data);
            this.layerSizes[this.layerSizes.length - 1] = SmallLanguageModelData.TOKENS.size();
            System.out.println(SmallLanguageModelData.TOKENS.size());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return data;
    }
}