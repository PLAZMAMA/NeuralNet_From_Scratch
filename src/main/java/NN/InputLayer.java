package NN;

public class InputLayer extends Layers{
    //instance variables
    double[] nodes;

    InputLayer(double[] input, int n_nodes){
        this.nodes = new double[n_nodes];
        for(int i = 0; i < n_nodes; i++){
            this.nodes[i] = input[i];
        }
        this.type = "Input";
    }

    public void calculate_nodes(double[] last_layer_vals, double[][] weights){
        System.out.println("this method doesn't do anything do to the input layer just being the input data shaped correctly which was done in the costructor");
    }
}