package io.github.betterclient.ai.training;

import java.util.List;
import java.util.Map;

public class TrainingDataPoint {
    public double cost;
    public List<NeuronPoint> inputLayer;

    public TrainingDataPoint(double cost, List<NeuronPoint> inputLayer) {
        this.cost = cost;
        this.inputLayer = inputLayer;
    }

    public static class NeuronPoint {
        public double bias;
        public double costGradientB;
        public Map<NeuronPoint, Double> weights;
        public Map<NeuronPoint, Double> costGradientW;

        public NeuronPoint(double bias, double costGradientB, Map<NeuronPoint, Double> weights, Map<NeuronPoint, Double> costGradientW) {
            this.bias = bias;
            this.weights = weights;
            this.costGradientB = costGradientB;
            this.costGradientW = costGradientW;
        }
    }
}
