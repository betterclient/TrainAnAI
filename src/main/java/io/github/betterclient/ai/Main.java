package io.github.betterclient.ai;

import io.github.betterclient.ai.object.Model;
import io.github.betterclient.ai.web.BackgroundAnimation;
import io.github.betterclient.ai.web.InfoButtonListener;
import io.github.betterclient.ai.web.ModelSelectionListener;
import io.github.betterclient.ai.web.TrainingStatus;

public class Main {
    public static ActivationFunction ACTIVATION_FUNCTION = ActivationFunction.SIGMOID;

    public static void main(String[] args) {
        InfoButtonListener.start();
        BackgroundAnimation.start();
        ModelSelectionListener.start();
        TrainingStatus.start();
    }

    public static Model trainModel() {
        Model model = ModelSelectionListener.createModel();
        if (model == null) return null;

        model.updateData();
        new Thread(model::train).start();

        return model;
    }
}