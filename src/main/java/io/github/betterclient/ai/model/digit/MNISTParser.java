package io.github.betterclient.ai.model.digit;

import io.github.betterclient.ai.training.TrainingInput;
import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;

/**
 * Parse the MNIST files generated by {@link io.github.betterclient.ai.MNISTParser}
 */
public class MNISTParser {
    public static Map<Integer, List<Map<Position, Boolean>>> PARSED = new HashMap<>();

    public static void parse() throws IOException {
        if (!PARSED.isEmpty()) return;

        for (int i = 0; i < 10; i++) {
            List<Map<Position, Boolean>> list = new ArrayList<>();

            InputStream stream = MNISTParser.class.getClassLoader().getResourceAsStream(i + ".json");
            if (stream == null) return;
            String src = new String(stream.readAllBytes());
            stream.close();

            for (String s : src.split("/")) {
                JSONObject object = new JSONObject(s);
                Map<Position, Boolean> map = new HashMap<>();
                for (String string : object.keySet()) {
                    int[] pos = Arrays.stream(string.split(",")).map(Integer::parseInt).mapToInt(Integer::intValue).toArray();
                    map.put(new Position(pos[0], pos[1]), object.getBoolean(string));
                }
                list.add(map);
            }
            PARSED.put(i, list);
        }

        System.out.println("Finished parsing MNIST");
    }

    public static List<TrainingInput> toSamples(int size) {
        List<TrainingInput> trainingInputs = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            float[] output = new float[10];
            Arrays.fill(output, 0);
            output[i] = 1;

            int index = 0;
            for (Map<Position, Boolean> map : PARSED.get(i)) {
                index++;
                if (index % 2 == 0) continue;
                if (index > (size / 2)) break;

                float[] input = new float[256];
                List<Float> list = map.values().stream().map(aBoolean -> aBoolean ? 1f : 0f).toList();
                for (int i1 = 0; i1 < list.size(); i1++) {
                    input[i1] = list.get(i1);
                }

                trainingInputs.add(new TrainingInput(input, output));
            }
        }

        return trainingInputs;
    }
}
