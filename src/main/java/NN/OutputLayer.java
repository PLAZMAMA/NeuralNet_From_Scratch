package NN;

import NN.Layers;
import java.lang.Double;

//class for the output layer of the neural network(only categorical with softmax for simplicity)
public class OutputLayer extends Layers{
    //instance variables
    Activations<Double[]> activation;
    Double[] nodes;


    //constructor method for the outputlayer class
    OutputLayer(Activations<Double[]> activation, int n_nodes){
        this.activation = activation;
        this.nodes = new Double[n_nodes];
    }

    @Override
    public void calculate_nodes(double[] last_layer_vals, double[][] weights){
        double sum;
        Double[] temp_nodes_vals = new Double[this.nodes.length];
        for(int column = 0; column < weights[0].length; column++){

            //gets the sum of the last layer's nodes vals times the corresponding/connected weights
            sum = 0.0;
            for(int row = 0; row < weights.length; row++){
                sum += last_layer_vals[row] * weights[row][column];
            }
            temp_nodes_vals[column] = new Double(sum);
        }
        this.nodes = this.activation.activate(temp_nodes_vals);
    }



}