import java.util.function.UnaryOperator;

public class NeuralNetwork {

    private double learningRate;
    private Layer[] layers;

    private UnaryOperator<Double> activation = Main.sigmoid;
    private UnaryOperator<Double> d_activation = Main.d_sigmoid;
    private UnaryOperator<Double> output_activation = Main.identity;
    private UnaryOperator<Double> d_output_activation = Main.d_identity;

    public NeuralNetwork(double learningRate_, int... sizes) {
        learningRate = learningRate_;
        layers = new Layer[sizes.length];
        for (int k = 0; k < sizes.length; k++) {
            int nextSize = 0;
            if (k < sizes.length - 1) nextSize = sizes[k + 1];

            layers[k] = new Layer(sizes[k], nextSize);

            for (int i = 0; i < sizes[k]; i++) {
                layers[k].biases[i] = Math.random() * 2.0 - 1.0;
                for (int j = 0; j < nextSize; j++) {
                    layers[k].weights[i][j] = Math.random() * 2.0 - 1.0;
                }
            }
        }
    }

    public double[] forwardPropagation(double[] inputs) {
        System.arraycopy(inputs, 0, layers[0].neurons, 0, inputs.length);
        for (int k = 1; k < layers.length; k ++) {
            for (int i = 0; i < layers[k].size; i ++) {
                layers[k].neurons[i] = layers[k].biases[i];
                for (int j = 0; j < layers[k - 1].size; j ++) {
                    layers[k].neurons[i] += layers[k - 1].neurons[j] * layers[k - 1].weights[j][i];
                }
                if (k != layers.length - 1)
                    layers[k].neurons[i] = activation.apply(layers[k].neurons[i]);
                else
                    layers[k].neurons[i] = output_activation.apply(layers[k].neurons[i]);
            }
        }
        return layers[layers.length - 1].neurons;
    }

    public void backwardPropagation(double[] targets) {
        Layer outputLayer = layers[layers.length - 1];
        double gradients[] = new double[outputLayer.size];
        for (int i = 0; i < outputLayer.size; i ++)
            gradients[i] = (outputLayer.neurons[i] - targets[i]) * d_activation.apply(outputLayer.neurons[i]);

        for (int k = layers.length - 2; k >= 0; k --) {
            Layer l1 = layers[k + 1], l = layers[k];

            double newGradients[] = new double[l.size];
            for (int i = 0; i < l.size; i ++) {
                newGradients[i] = 0;
                for (int j = 0; j < l1.size; j ++) {
                    newGradients[i] += gradients[j] * l.weights[i][j];
                }
            }

            double[][] newWeights = new double[l.size][l1.size];
            for (int i = 0; i < l.size; i ++) {
                for (int j = 0; j < l1.size; j ++) {
                    l.weights[i][j] -= learningRate * l.neurons[i] * gradients[j];
                }
            }

            for (int j = 0; j < l1.size; j ++) {
                l1.biases[j] -= learningRate * gradients[j];
            }

            l.weights = newWeights;
            gradients = newGradients;
        }
    }

    void train() {

    }
}