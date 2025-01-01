package io.github.betterclient.ai.web;

import io.github.betterclient.ai.Main;
import io.github.betterclient.ai.object.Model;
import org.teavm.jso.browser.Window;
import org.teavm.jso.dom.html.HTMLDocument;
import org.teavm.jso.dom.html.HTMLElement;

public class TrainingStatus {
    public static Model TRAINED_MODEL = null;

    public static void start() {
        HTMLDocument document = HTMLDocument.current();

        HTMLElement container = document.getElementById("container");
        container.addEventListener("change", evt -> {
            HTMLElement target = (HTMLElement) evt.getTarget();
            if (target != null && target.getClassName().contains("TRAINING_RESET")) {
                untrain();
            }
        });

        document.getElementById("TRAIN").onClick(evt -> train());

        HTMLDocument.current().getElementById("GUESS").onClick(evt -> {
            if (TRAINED_MODEL == null) return;

            HTMLDocument.current().getElementById("MODEL_OUTPUT").setInnerText(TRAINED_MODEL.getOutput());
        });
    }

    private static void train() {
        HTMLDocument document = HTMLDocument.current();

        HTMLElement trainingStatus = document.getElementById("TRAINING_STATUS");
        trainingStatus.setInnerText("Training");
        trainingStatus.getStyle().setProperty("color", "yellow");

        //Train the model 2 frames later, to let the training text display
        Window.requestAnimationFrame(timestamp -> Window.requestAnimationFrame(timestamp1 -> {
            TRAINED_MODEL = Main.trainModel();

            if (TRAINED_MODEL != null) {
                trainingStatus.setInnerText("Trained");
                trainingStatus.getStyle().setProperty("color", "green");
            } else {
                untrain();
            }
        }));
    }

    private static void untrain() {
        HTMLDocument document = HTMLDocument.current();

        HTMLElement trainingStatus = document.getElementById("TRAINING_STATUS");
        trainingStatus.setInnerText("Not trained");
        trainingStatus.getStyle().setProperty("color", "red");
    }
}