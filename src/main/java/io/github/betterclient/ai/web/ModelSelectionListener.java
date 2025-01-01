package io.github.betterclient.ai.web;

import io.github.betterclient.ai.model.digit.DigitRecognitionModel;
import io.github.betterclient.ai.model.IsBiggerThan100Model;
import io.github.betterclient.ai.object.Model;
import org.teavm.jso.dom.html.HTMLDocument;
import org.teavm.jso.dom.html.HTMLElement;
import org.teavm.jso.dom.html.HTMLSelectElement;

public class ModelSelectionListener {
    public static void start() {
        HTMLSelectElement selectElement = (HTMLSelectElement) HTMLDocument.current().getElementById("selectmodel");
        onChange(); //Initial.

        selectElement.addEventListener("change", evt -> onChange());
    }

    public static Model createModel() {
        HTMLSelectElement selectElement = (HTMLSelectElement) HTMLDocument.current().getElementById("selectmodel");
        String modelValue = selectElement.getValue();

        return switch (modelValue) {
            case "isbiggerthan100" -> new IsBiggerThan100Model();
            case "digitrecog" -> new DigitRecognitionModel();
            default -> null;
        };
    }

    private static void onChange() {
        HTMLSelectElement selectElement = (HTMLSelectElement) HTMLDocument.current().getElementById("selectmodel");
        String newValue = selectElement.getValue();

        HTMLElement element = HTMLDocument.current().getElementById("MODEL_SPECIFIC");
        element.setInnerHTML(""); //clear

        HTMLElement element0 = HTMLDocument.current().getElementById("MODEL_SPECIFIC_INPUT");
        element0.setInnerHTML(""); //clear

        switch (newValue) {
            case "isbiggerthan100":
                IsBiggerThan100Model.appendSettings(element);
                IsBiggerThan100Model.appendInput(element0);
                break;
            case "digitrecog":
                DigitRecognitionModel.appendSettings(element);
                DigitRecognitionModel.appendInput(element0);
                break;
            default:
                break;
        }
    }
}