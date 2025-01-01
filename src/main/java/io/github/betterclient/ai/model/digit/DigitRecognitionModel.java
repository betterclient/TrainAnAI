package io.github.betterclient.ai.model.digit;

import io.github.betterclient.ai.object.Model;
import io.github.betterclient.ai.training.TrainingInput;
import org.teavm.jso.canvas.CanvasRenderingContext2D;
import org.teavm.jso.dom.html.HTMLButtonElement;
import org.teavm.jso.dom.html.HTMLCanvasElement;
import org.teavm.jso.dom.html.HTMLElement;
import org.teavm.jso.dom.html.TextRectangle;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

public class DigitRecognitionModel extends Model {
    private static final Map<Position, Boolean> DRAWING = new HashMap<>();

    public DigitRecognitionModel() {
        super("DigitRecognition", "description", 10, 0.01f, 100, new int[] {256, 16, 16, 10});

        try {
            MNISTParser.parse();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void appendSettings(HTMLElement element) {

    }

    public static void appendInput(HTMLElement element) {
        clear();

        HTMLCanvasElement canvas = (HTMLCanvasElement) element.getOwnerDocument().createElement("canvas");
        CanvasRenderingContext2D context2D = (CanvasRenderingContext2D) canvas.getContext("2d");
        canvas.setWidth(16);
        canvas.setHeight(16);

        float scale = 15;
        canvas.getStyle().setProperty("width", 16 * scale + "px");
        canvas.getStyle().setProperty("height", 16 * scale + "px");

        context2D.setFillStyle("black");
        context2D.fillRect(0, 0, 16, 16);

        AtomicBoolean isPainting = new AtomicBoolean(false);
        canvas.onMouseUp(evt -> isPainting.set(false));
        canvas.onMouseDown(evt -> isPainting.set(true));
        canvas.onMouseMove(evt -> {
            if (!isPainting.get()) return;

            TextRectangle rect = canvas.getBoundingClientRect();
            double x = Math.floor((double) (evt.getClientX() - rect.getLeft()) / scale);
            double y = Math.floor((double) (evt.getClientY() - rect.getTop()) / scale);

            context2D.setFillStyle("white");
            context2D.fillRect(x, y, 1, 1);
            DRAWING.put(new Position((int) x, (int) y), true);
        });

        HTMLButtonElement button = (HTMLButtonElement) element.getOwnerDocument().createElement("button");
        button.onClick(evt -> {
            context2D.setFillStyle("black");
            context2D.fillRect(0, 0, 16, 16);
            clear();
        });
        button.setInnerText("Clear");

        element.appendChild(button);

        element.appendChild(element.getOwnerDocument().createElement("br"));

        element.appendChild(canvas);
    }

    private static void clear() {
        DRAWING.clear();
        for (int x = 0; x < 16; x++) {
            for (int y = 0; y < 16; y++) {
                DRAWING.put(new Position(x, y), false);
            }
        }
    }

    @Override
    public void updateData() {
        int hiddenLayers = getSlider("hiddenlayers");

        this.epochs = getSlider("epochs");
        this.learningRate = getSlider("learningRate") / 10000f;
        this.trainingSampleSize = getSlider("trainingsamples");

        this.layerSizes = new int[hiddenLayers + 2];
        for (int i = 0; i < this.layerSizes.length; i++) {
            if (i == 0) this.layerSizes[i] = 16;
            else if (i == this.layerSizes.length - 1) this.layerSizes[i] = 256;
            else this.layerSizes[i] = 10;
        }
    }

    @Override
    public String getInputForData(String data) {
        float[] v = new float[256];
        List<Float> floats = DRAWING.values().stream().map(aBoolean -> aBoolean ? 1f : 0f).toList();
        for (int i = 0; i < floats.size(); i++) {
            v[i] = floats.get(i);
        }

        float[] out = this.network.forward(v);

        int hi = 0;
        float hg = 0;
        for (int i = 0; i < out.length; i++) {
            if (out[i] > hg) {
                hg = out[i];
                hi = i;
            }
        }

        return "Model Guess: " + hi;
    }

    @Override
    public String getOutput() {
        return getInputForData("");
    }

    @Override
    public List<TrainingInput> getTrainingSamples() {
        return MNISTParser.toSamples(this.trainingSampleSize);
    }
}
