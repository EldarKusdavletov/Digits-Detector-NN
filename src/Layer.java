public class Layer {

    public int size;
    public double[] gradient;
    public double[] neurons;
    public double[] biases;
    public double[][] weights;

    public Layer(int size_, int sizeNext) {
        size = size_;
        gradient = new double[size_];
        neurons = new double[size_];
        biases = new double[size_];
        weights = new double[size_][sizeNext];
    }
}
