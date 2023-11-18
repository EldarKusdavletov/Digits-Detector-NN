import java.util.function.UnaryOperator;

public class NeuralNetwork {

    private double learningRate;
    private Layer[] layers;
    private UnaryOperator<Double> sigmoid = x -> 1.0 / (1.0 + Math.exp(-x));
    private UnaryOperator<Double> d_sigmoid = x -> x * (1.0 - x);

    public NeuralNetwork(double learningRate_, int... sizes) {
        learningRate = learningRate_;
        layers = new Layer[sizes.length];
        for (int k = 0; k < sizes.length; k ++) {
            int nextSize = 0;
            if (k < sizes.length - 1) nextSize = sizes[k + 1];

            layers[k] = new Layer(sizes[k], nextSize);

            for (int i = 0; i < sizes[k]; i ++) {
                layers[k].biases[i] = Math.random() * 2.0 - 1.0;
                for (int j = 0; j < nextSize; j ++) {
                    layers[k].weights[i][j] = Math.random() * 2.0 - 1.0;
                }
            }
        }
    }

    public double[] forwardPropagation(double[] inputs) {
        layers[0].neurons = inputs;

        for (int k = 1; k < layers.length; k ++) {
            Layer l = layers[k - 1], l1 = layers[k];

            for (int j = 0; j < l1.size; j ++) {
                l1.neurons[j] = l1.biases[j];
                for (int i = 0; i < l.size; i++) {
                    l1.neurons[j] += l.neurons[i] * l.weights[i][j];
                }

                l1.neurons[j] = sigmoid.apply(l1.neurons[j]);
            }
        }
        return layers[layers.length - 1].neurons;
    }

    public void backwardPropagation(double[] targets) {
        Layer outputLayer = layers[layers.length - 1];
        for (int i = 0; i < outputLayer.size; i ++) {
            outputLayer.gradient[i] = learningRate * (outputLayer.neurons[i] - targets[i]);
        }

        for (int k = layers.length - 1; k > 0; k --) {
            Layer l = layers[k - 1], l1 = layers[k];

            for (int i = 0; i < l.size; i ++) {
                l.gradient[i] = 0;
                for (int j = 0; j < l1.size; j ++) {
                    l.gradient[i] += l1.gradient[j] * l.weights[i][j];
                    l.weights[i][j] -= l1.gradient[j] * d_sigmoid.apply(l1.neurons[j])  * l.neurons[i];
                }
            }

            for (int j = 0; j < l1.size; j++) {
                l1.biases[j] -= l1.gradient[j] * d_sigmoid.apply(l1.neurons[j]);
            }
        }

    }

    void train(double[] inputs, double[] targets) {

    }
}