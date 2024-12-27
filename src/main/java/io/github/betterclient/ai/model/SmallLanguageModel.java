package io.github.betterclient.ai.model;

import io.github.betterclient.ai.object.Model;
import io.github.betterclient.ai.training.TrainingInput;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class SmallLanguageModel extends Model {
    public SmallLanguageModel() {
        super(
                "SLM - Small language model",
                "A really small language model that you shouldn't expect to make sense",
                200,
                0.1f,
                0.1f,
                100, //you shouldn't be able to adjust this
                new int[] {64, 64}
        );
    }

    @Override
    public void updateData() {}

    @Override
    public String getInputForData(String data) {
        float[] modelOutput = network.forward(reverseDisplay(data));
        return display(modelOutput);
    }

    @Override
    public List<TrainingInput> getTrainingSamples() {
        List<TrainingInput> data = new ArrayList<>();

        File f = new File("slm.txt");
        try {
            Scanner scanner = new Scanner(f);
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();

                data.add(new TrainingInput(
                        reverseDisplay(line.split(": ")[0]),
                        reverseDisplay(line.split(": ")[1])
                ));
            }
            scanner.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return data;
    }

    //----------------UTIL-------------

    public static String display(float[] normalizedArray) {
        StringBuilder builder = new StringBuilder();

        for (float v : normalizedArray) {
            if (v == 0) {
                return builder.toString();
            }

            builder.append((char) (v * 128));
        }

        return builder.toString();
    }

    public static float[] reverseDisplay(String input) {
        char[] charArray = input.toCharArray();
        float[] reverseDisplayed = new float[64];

        //Fill array with 0s
        Arrays.fill(reverseDisplayed, 0);
        int index = 0;
        for (char c : charArray) {
            reverseDisplayed[index] = c / 128f;
            index++;
        }
        return reverseDisplayed;
    }
}