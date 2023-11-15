import java.util.function.UnaryOperator;

public class Main {

    static UnaryOperator<Double> sigmoid = x -> 1.0 / (1.0 + Math.exp(-x));
    static UnaryOperator<Double> d_sigmoid = y -> y * (1.0 - y);
    static UnaryOperator<Double> identity = x -> x;
    static UnaryOperator<Double> d_identity = x -> 1.0;

    public static void main(String[] args) {
        digitsRun();
    }

    public static void digitsRun() {

    }

}