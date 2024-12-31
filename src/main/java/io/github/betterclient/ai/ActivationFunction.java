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
    ),

    //Relu is too fast to cache
    RELU(ActivationFunction::relu, ActivationFunction::reluDerivative);

    public final Function<Float, Float> func, derivative;
    ActivationFunction(Function<Float, Float> func, Function<Float, Float> derivative) {
        this.func = func;
        this.derivative = derivative;
    }

    static final Map<Float, Float> normalCache = new HashMap<>();
    static final Map<Float, Float> derivativeCache = new HashMap<>();

    public static float cacheOrCreate(float x, Function<Float, Float> func, boolean isDerivative) {
        return (isDerivative ? derivativeCache : normalCache).computeIfAbsent(x, func);
    }

    public static float relu(float x) {
        return Math.max(0, x);
    }

    public static float reluDerivative(float x) {
        return x > 0 ? 1 : 0;
    }

    public static float geluDerivative(float x) {
        float sqrt2OverPi = (float) Math.sqrt(2f / Math.PI);
        float coefficient = 0.044715F;

        float term = (float) (x + coefficient * Math.pow(x, 3));
        float tanhZ = (float) Math.tanh(sqrt2OverPi * term);
        float sech2Z = 1 - tanhZ * tanhZ; // sech^2(z) = 1 - tanh^2(z)

        float zDerivative = (float) (sqrt2OverPi * (1f + 3f * coefficient * Math.pow(x, 2f)));

        return 0.5f * (1 + tanhZ) + 0.5f * x * sech2Z * zDerivative;
    }

    public static float gelu(float x) {
        float sqrt2OverPi = (float) Math.sqrt(2 / Math.PI);
        float coefficient = 0.044715f;

        float term = (float) (x + coefficient * Math.pow(x, 3f));
        return (float) (0.5f * x * (1f + Math.tanh(sqrt2OverPi * term)));
    }

    private static float sigmoid(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }

    private static float sigmoidDerivative(float x) {
        float sigmoid = sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }
}
