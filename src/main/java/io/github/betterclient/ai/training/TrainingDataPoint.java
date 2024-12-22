package io.github.betterclient.ai.training;

import java.util.List;
import java.util.Map;

public record TrainingDataPoint(double cost, List<NeuronPoint> inputLayer) {
    public record NeuronPoint(double bias, Map<NeuronPoint, Double> weights) {}
}
