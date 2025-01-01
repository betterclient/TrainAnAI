package io.github.betterclient.ai.web;

import org.teavm.jso.browser.Window;
import org.teavm.jso.dom.html.HTMLBodyElement;
import org.teavm.jso.dom.html.HTMLDocument;

public class BackgroundAnimation {
    public static void start() {
        Window.requestAnimationFrame(BackgroundAnimation::animate);
    }

    private static double degree = 0;
    private static double degree1 = 0;
    private static double degree2 = 0;
    private static double degree3 = 0;

    private static void animate(double timestamp) {
        HTMLDocument document = HTMLDocument.current();
        HTMLBodyElement body = document.getBody();

        body.getStyle().setProperty(
                "background-image",
                "linear-gradient(" + (degree = degree+.25) + "deg, blue, #37aed0)"
        );

        document.getElementById("d1").getStyle().setProperty(
                "background-image",
                "linear-gradient(" + (degree1 = degree1-.75) + "deg, #37aed0, #38bfe1)"
        );
        document.getElementById("d2").getStyle().setProperty(
                "background-image",
                "linear-gradient(" + (degree2--) + "deg, #37aed0, #38bfe1)"
        );
        document.getElementById("d3").getStyle().setProperty(
                "background-image",
                "linear-gradient(" + (degree3 = degree3-.50) + "deg, #37aed0, #38bfe1)"
        );

        if (degree > 360) degree = 0;
        if (degree1 < -360) degree1 = 0;
        if (degree2 < -360) degree2 = 0;
        if (degree3 < -360) degree3 = 0;

        Window.requestAnimationFrame(BackgroundAnimation::animate);
    }
}
