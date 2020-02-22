package NN;

abstract class Layers{
    //instance variable
    String type;
    double[] nodes;
    double[] biases;
    Activations<Double> activation;

    //abstract methods
    public abstract void calculate_nodes(double[] last_layer_vals, double[][] weights);
}