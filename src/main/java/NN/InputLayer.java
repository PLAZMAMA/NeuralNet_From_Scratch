package NN;

/*
*The InputLayer class was constructed to allow easier type manegment in while creating the Model class.
*This allows easier type manegment because, it allows the Model class to use the
*inputlayer(aka the data) as a Layers class/type.
*/
public class InputLayer extends Layers{

    InputLayer(int n_nodes){
        super.nodes = new double[n_nodes];
        super.type = "Input";
    }

    //this method doesn't do anything since the input layer is just the data.
    public void calculate_nodes(double[] last_layer_vals, double[][] weights){
        System.out.println("this method doesn't do anything do to the input layer just being the input data shaped correctly which was done in the costructor");
    }
}