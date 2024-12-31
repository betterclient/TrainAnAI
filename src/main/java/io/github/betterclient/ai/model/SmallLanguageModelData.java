package io.github.betterclient.ai.model;

import io.github.betterclient.ai.training.TrainingInput;
import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class SmallLanguageModelData {
    public static Set<String> TOKENS = new HashSet<>();

    public static void parse(List<TrainingInput> to) throws IOException {
        TOKENS.clear();
        TOKENS.add("<start>");

        InputStream stream = SmallLanguageModelData.class.getClassLoader().getResourceAsStream("slm_data.json");
        byte[] read = stream.readAllBytes();
        stream.close();

        JSONObject object = new JSONObject(new String(read));

        //First loop to append all tokens
        int i11 = 0;
        for (String ngrams : object.getJSONObject("ngrams").keySet()) {
            i11++;
            if(i11 % 2 == 0) continue;
            String[] split = ngrams.substring(1, ngrams.length() - 1).split(", ");
            for (String s : split) {
                if(s.length() > 3) s = s.substring(1, s.length() - 1);
                if (s.charAt(0) == '"' && s.length() > 3) s = s.substring(1, s.length() - 1);

                TOKENS.add(s.toLowerCase());
            }
            JSONObject object1 = object.getJSONObject("ngrams").getJSONObject(ngrams);
            for (String s : object1.keySet()) {
                if(s.length() > 3) s = s.substring(1, s.length() - 1);
                if (s.charAt(0) == '"' && s.length() > 3) s = s.substring(1, s.length() - 1);
                TOKENS.add(s.toLowerCase());
            }
        }
        TOKENS.add("<end>");

        //Second loop to generate all data
        i11 = 0;
        for (String ngrams : object.getJSONObject("ngrams").keySet()) {
            i11++;
            if(i11 % 2 == 0) continue;
            float[] inputs = new float[3];

            String[] split = ngrams.substring(1, ngrams.length() - 1).split(", ");
            int index = 0;
            for (String s : split) {
                if(s.length() > 3) s = s.substring(1, s.length() - 1);
                if (s.charAt(0) == '"' && s.length() > 3) s = s.substring(1, s.length() - 1);

                inputs[index] = TOKENS.stream().toList().indexOf(s.toLowerCase());
                index++;
            }
            JSONObject object1 = object.getJSONObject("ngrams").getJSONObject(ngrams);
            for (String s : object1.keySet()) {
                int a = object1.getInt(s);
                if (s.charAt(0) == '"' && s.length() > 3) s = s.substring(1, s.length() - 1);

                for (int i1 = 0; i1 < a; i1++) {
                    float[] outputs = new float[TOKENS.size()];
                    int wantedIndex = TOKENS.stream().toList().indexOf(s.toLowerCase());

                    for (int i = 0; i < outputs.length; i++) {
                        if (i == wantedIndex) {
                            outputs[i] = 0;
                        } else {
                            outputs[i] = 1;
                        }
                    }

                    to.add(new TrainingInput(inputs, outputs));
                }
            }
        }
    }

    public static float tokenize(String s) {
        return Math.max(0, TOKENS.stream().toList().indexOf(s.toLowerCase()));
    }

    public static String untokenize(float[] forward) {
        int lowestChanceIndex = 0;
        float lowestChance = 0;

        int index = 0;
        for (float v : forward) {
            if (v < lowestChance) {
                lowestChanceIndex = index;
                lowestChance = v;
            }

            index++;
        }

        return TOKENS.stream().toList().get(lowestChanceIndex);
    }
}