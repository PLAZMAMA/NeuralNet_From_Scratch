package NN;

abstract class Layers{
    public abstract void calculate_nodes(double[] last_layer_vals, double[][] weights);
}