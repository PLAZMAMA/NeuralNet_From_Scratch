package NN;

public class InputLayer extends Layers{
    //instance variables
    double[] nodes;

    InputLayer(double[] input){
        this.nodes = new double[input.length];
        for(int i = 0; i < input.length; i++){
            this.nodes[i] = input[i];
        }
        this.type = "Input";
    }

    public void calculate_nodes(double[] last_layer_vals, double[][] weights){
        System.out.println("this method doesn't do anything do to the input layer just being the input data shaped correctly which was done in the costructor");
    }
}