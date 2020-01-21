package NN;
import java.lang.reflect.Method;

import NN.Activations;

/*
* this class creates dense neural net layer
*/
public class Dense{
    //instance variables
    double[] biases;
    double[][] nodes;
    Method activation;

    /*
    * constructor method for the Dense class
    */
    public void Dense(int n_nodes, String activation){
        this.nodes = new double[n_nodes][n_nodes];
        this.biases = new double[n_nodes];
        this.activation = 
        for(int i = 0; i < n_nodes; i++){
            this.biases[i] = 0.0;
        }


    }
    
    /*
    * calculate the layer's nodes with the given weights, last layers nodes and biases.
    * it sums up all the given weights and biases, then adds the bias, lastly it puts it through the activation function 
    */
    public double[] calculate_nodes(double[] last_layer_vals, double[][] weights){
        for(int i = 0; i < weights.length; i++){
            for(int j = 0; j < last_layer_vals.length; i++){

            }
        }
        return(this.nodes);
    }

    

}