package NN;

import NN.Activations;

//this class creates dense neural net layer
public class Dense extends Layers{
    //instance variables
    double bias;
    double[] nodes;
    Activations<Double> activation;
    

    
    //constructor for the Dense class
    Dense(int n_nodes, Activations<Double> activation){
        this.nodes = new double[n_nodes];
        this.bias = 0.0;
        this.activation = activation;


    }
    
    /*
    calculate the layer's nodes with the given weights and last layers nodes.
    it sums up all the given weights and biases, then adds the bias, lastly it puts it through the activation function .
    the function cahnges this.nodes to the nodes with the activation function added
    */
    public void calculate_nodes(double[] last_layer_vals, double[][] weights){
        double sum;
        for(int columb = 0; columb < weights[0].length; columb++){

            //gets the sum of the last layer's nodes vals times the corresponding/connected weight
            sum = 0.0;
            for(int row = 0; row < weights.length; row++){
                sum += last_layer_vals[row] * weights[row][columb];
            }

            //adds the bias to the sum, puts in through the given(in the constructor) activation function and stores it in this.nodes
            this.nodes[columb] = this.activation.activate(sum + this.bias);
        }
    }

    

}