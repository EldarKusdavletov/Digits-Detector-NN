public class Layer {

    public int size;
    public double[] neurons;
    public double[] biases;
    public double[][] weights;

    public Layer(int size_, int nextSize_) {
        size = size_;
        neurons = new double[size_];
        biases = new double[size_];
        weights = new double[size_][nextSize_];
    }
}
