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
        System.arraycopy(inputs, 0, layers[0].neurons, 0, inputs.length);
        for (int k = 1; k < layers.length; k++) {
            for (int i = 0; i < layers[k].size; i++) {
                layers[k].neurons[i] = layers[k].biases[i];
                for (int j = 0; j < layers[k - 1].size; j++) {
                    layers[k].neurons[i] += layers[k - 1].neurons[j] * layers[k - 1].weights[j][i];
                }

                layers[k].neurons[i] = activation.apply(layers[k].neurons[i]);
            }
        }
        return layers[layers.length - 1].neurons;
    }

    /*public void backwardPropagation(double[] targets) {
        Layer outputLayer = layers[layers.length - 1];

        double[] errors = new double[outputLayer.size];
        double gradients[] = new double[outputLayer.size];

        for (int i = 0; i < outputLayer.size; i++) {
            gradients[i] = (outputLayer.neurons[i] - targets[i]) * d_activation.apply(outputLayer.neurons[i]);
            errors[i] = (outputLayer.neurons[i] - targets[i]);
            // gradients[i] = errors[i] * d_activation.apply(outputLayer.neurons[i]);
        }

        for (int k = layers.length - 2; k >= 0; k--) {
            Layer l1 = layers[k + 1], l = layers[k];
            double[] errorsNext = new double[l.size];
            double newGradients[] = new double[l.size];

            for (int i = 0; i < l.size; i++) {
                errorsNext[i] = 0;
                newGradients[i] = 0;
                for (int j = 0; j < l1.size; j++) {
                    errorsNext[i] += l.weights[i][j] * errors[j];
                    newGradients[i] += l.weights[i][j] * gradients[j] * (d_activation.apply(layers[k].neurons[i]) / d_activation.apply(layers[k + 1].neurons[j]));
                }
            }

            // gradients[i] = errors[i] * d_activation.apply(layers[k + 1].neurons[i]);

            for (int i = 0; i < l1.size; i++) {
                for (int j = 0; j < l.size; j++) {
                    // l.weights[j][i] += learningRate * l.neurons[j] * errors[i] * d_activation.apply(layers[k + 1].neurons[i]);
                    l.weights[j][i] += learningRate * l.neurons[j] * gradients[i]; // errors[i] * d_activation.apply(outputLayer.neurons[i]);
                }
            }

            for (int i = 0; i < l1.size; i++) {
                // l1.biases[i] += learningRate * errors[i] * d_activation.apply(layers[k + 1].neurons[i]);
                l1.biases[i] += learningRate * gradients[i]; // errors[i] * d_activation.apply(outputLayer.neurons[i]);
            }

            gradients = newGradients;
            errors = errorsNext;
        }
    }*/

    public void backwardPropagation(double[] targets) {
        Layer outputLayer = layers[layers.length - 1];
        double errors[] = new double[outputLayer.size];

        for (int i = 0; i < outputLayer.size; i++)
            errors[i] = outputLayer.neurons[i] - targets[i];

        for (int k = layers.length - 2; k >= 0; k--) {
            Layer l = layers[k], l1 = layers[k + 1];

            double[] newErrors = new double[l.size];
            for (int i = 0; i < l.size; i++) {
                newErrors[i] = 0;
                for (int j = 0; j < l1.size; j++) {
                    newErrors[i] += l.weights[i][j] * errors[j];
                }
                newErrors[i] *= l.neurons[i] * (1 - l.neurons[i]);
            }

            for (int i = 0; i < l.size; i++) {
                for (int j = 0; j < l1.size; j++) {
                    l.weights[i][j] -= learningRate * errors[j] * l.neurons[i];
                }
            }

            for (int j = 0; j < l1.size; j++) {
                l1.biases[j] -= learningRate * errors[j];
            }

            errors = newErrors;
        }
    }

    void train() {

    }
}