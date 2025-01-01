package io.github.betterclient.ai.web;

import org.teavm.jso.dom.html.HTMLDocument;
import org.teavm.jso.dom.html.HTMLElement;

public class InfoButtonListener {
    private static HTMLElement infobox;

    public static void start() {
        HTMLDocument doc = HTMLDocument.current();

        infobox = doc.getElementById("infobox");
        doc.getBody().removeChild(infobox);

        doc.getElementById("info-epochs").onClick(evt -> setInfoBox(
                """
                <h3>Epochs</h3>
                Amount of times to train.
                <br>
                <br>
                Precision Affect: <span style="color: orange;">High</span>
                <br>
                Training time affect: <span style="color: orange;">High</span>
                <br>
                Higher is better.
                """));

        doc.getElementById("info-learningrate").onClick(evt -> setInfoBox(
                """
                <h3>Learning Rate</h3>
                The slope algorithm amount for the model.
                <br>
                <br>
                Precision Affect: <span style="color: orange;">High</span>
                """));

        doc.getElementById("info-hiddenlayers").onClick(evt -> setInfoBox(
                """
                <h3>Hidden Layers</h3>
                Amount of layers between input and output.
                <br>
                <br>
                Precision Affect: <span style="color: yellow;">Depends</span>
                <br>
                Training Time Affect: <span style="color: red;">Extreme</span>
                """));

        doc.getElementById("info-sample").onClick(evt -> setInfoBox(
                """
                <h3>Sample Amounts</h3>
                Amount of training samples.
                <br>
                <br>
                Precision Affect: <span style="color: yellow;">Depends</span>
                <br>
                Training Time Affect: <span style="color: orange;">Extreme+</span>
                """));
    }

    private static void setInfoBox(String s) {
        HTMLDocument doc = HTMLDocument.current();
        if (doc.getElementById("infobox") == null) doc.getBody().appendChild(infobox);

        doc.getElementById("infobox").setInnerHTML(s.trim());
    }
}
