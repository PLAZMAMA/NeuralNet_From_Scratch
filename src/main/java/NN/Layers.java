package NN;

abstract class Layers{
    //instance variable
    String type;
    double[] nodes;
    double[] biases;

    //abstract methods
    public abstract void calculate_nodes(double[] last_layer_vals, Double[][] weights);
}