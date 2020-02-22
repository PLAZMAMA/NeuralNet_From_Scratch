package NN;

import java.util.Arrays;

import NN.Activations;

//this class creates dense neural net layer
public class Dense extends Layers{
    //instance variables
    Activations<Double> activation;
    
    //constructor for the Dense class
    Dense(Activations<Double> activation,  int n_nodes){
        super.biases = new double[n_nodes];
        Arrays.fill(super.biases, 0.0);
        super.nodes = new double[n_nodes];
        this.activation = activation;
        super.type = "Dense";
    }
    
    /*
    calculate the layer's nodes with the given weights and last layers nodes.
    it sums up all the given weights and biases, then adds the bias, lastly it puts it through the activation function .
    the function cahnges this.nodes to the nodes with the activation function added
    */
    public void calculate_nodes(double[] last_layer_vals, double[][] weights){
        double sum;
        for(int row = 0; row < weights.length; row++){

            //gets the sum of the last layer's nodes vals times the corresponding/connected weights
            sum = 0.0;
            for(int column = 0; column < weights[row].length; column++){
                sum += last_layer_vals[column] * weights[row][column];
            }

            //adds the bias to the sum, puts in through the given(in the constructor) activation function and stores it in this.nodes
            super.nodes[row] = this.activation.activate(sum + super.biases[row]);
        }
    }

    

}