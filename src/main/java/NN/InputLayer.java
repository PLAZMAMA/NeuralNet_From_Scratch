package NN;

/*
*The InputLayer class was constructed to allow easier type manegment in while creating the Model class.
*This allows easier type manegment because, it allows the Model class to use the
*inputlayer(aka the data) as a Layers class/type.
*/
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

    //this method doesn't do anything since the input layer is just the data.
    public void calculate_nodes(double[] last_layer_vals, double[][] weights){
        System.out.println("this method doesn't do anything do to the input layer just being the input data shaped correctly which was done in the costructor");
    }
}