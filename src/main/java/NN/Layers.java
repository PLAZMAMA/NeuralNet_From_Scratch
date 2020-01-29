package NN;

abstract class Layers{
    //instance variable
    String type;
    double[] nodes;

    //abstract methods
    public abstract void calculate_nodes(double[] last_layer_vals, double[][] weights);
}