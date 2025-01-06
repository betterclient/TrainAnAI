package io.github.betterclient.ai;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * I have no idea how any of this works
 */
public enum ActivationFunction {
    SIGMOID(
            x -> cacheOrCreate(x, ActivationFunction::sigmoid, false),

            x -> cacheOrCreate(x, ActivationFunction::sigmoidDerivative, true)
    ),

    GELU(
            x -> cacheOrCreate(x, ActivationFunction::gelu, false),

            x -> cacheOrCreate(x, ActivationFunction::geluDerivative, true)
    );

    public final Function<Double, Double> func, derivative;
    ActivationFunction(Function<Double, Double> func, Function<Double, Double> derivative) {
        this.func = func;
        this.derivative = derivative;
    }

    static final Map<Double, Double> normalCache = new HashMap<>();
    static final Map<Double, Double> derivativeCache = new HashMap<>();

    public static double cacheOrCreate(double x, Function<Double, Double> func, boolean isDerivative) {
        return (isDerivative ? derivativeCache : normalCache).computeIfAbsent(x, func);
    }

    public static double geluDerivative(double x) {
        double sqrt2OverPi = (float) Math.sqrt(2f / Math.PI);
        double coefficient = 0.044715F;

        double term = -x + coefficient * Math.pow(x, 3);
        double tanhZ = -Math.tanh(sqrt2OverPi * term);
        double sech2Z = 1 - tanhZ * tanhZ; // sech^2(z) = 1 - tanh^2(z)

        double zDerivative = (float) (sqrt2OverPi * (1f + 3f * coefficient * Math.pow(x, 2f)));

        return 0.5f * (1 + tanhZ) + 0.5f * x * sech2Z * zDerivative;
    }

    public static double gelu(double x) {
        double sqrt2OverPi = Math.sqrt(2 / Math.PI);
        double coefficient = 0.044715f;

        double term = (x + coefficient * Math.pow(x, 3f));
        return (0.5f * x * (1f + Math.tanh(sqrt2OverPi * term)));
    }

    private static double sigmoid(double x) {
        return (1 / (1 + Math.exp(-x)));
    }

    private static double sigmoidDerivative(double x) {
        double sigmoid = ActivationFunction.SIGMOID.func.apply(x); //Allow caching on derivative too
        return sigmoid * (1 - sigmoid);
    }

    public void reset() {
        normalCache.clear();
        derivativeCache.clear();
    }
}