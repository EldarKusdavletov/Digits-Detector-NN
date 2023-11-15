import java.util.function.UnaryOperator;

public class NeuralNetwork {

    private double learningRate;
    private Layer[] layers;
    private UnaryOperator<Double> activation;

    private double dot(double[] a, double[] b) {
        assert(a.length == b.length);

        double res = 0;
        for (int i = 0; i < a.length; i ++) {
            res += a[i] * b[i];
        }

        return res;
    }

    public double[] forwardPropagation(double[] inputs) {
        System.arraycopy(inputs, 0, layers[0].neurons, 0, inputs.length);
        for (int i = 1; i < layers.length; i ++) {
            for (int j = 0; j < layers[i].size; j ++) {
                layers[i].neurons[j] = layers[i].biases[j];
                for (int k = 0; k < layers[i - 1].size; k ++) {
                    layers[i].neurons[j] += layers[i - 1].neurons[k] * layers[i - 1].weights[k][j];
                }
                layers[i].neurons[j] = activation.apply(layers[i].neurons[j]);
            }
        }
        return layers[layers.length - 1].neurons;
    }
}