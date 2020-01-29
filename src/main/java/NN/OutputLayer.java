package NN;

import NN.Layers;
import java.lang.Double;

//class for the output layer of the neural network(only categorical with softmax for simplicity)
public class OutputLayer extends Layers{
    //instance variables
    Activations<Double[]> activation;
    double[] nodes;



    //constructor method for the outputlayer class
    OutputLayer(Activations<Double[]> activation, int n_nodes){
        this.activation = activation;
        this.nodes = new double[n_nodes];
        this.type = "Output";
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
            //stores the sum of each node in a temporary Double array
            temp_nodes_vals[column] = new Double(sum);
        }
        //the temporary Double array is put into a activation function(in this case probably softmax),
        //then the output of the activation function is stored in this.nodes/nodes by dumping the temp array into the 
        temp_nodes_vals = this.activation.activate(temp_nodes_vals);
        for(int i = 0; i < temp_nodes_vals.length; i++){
            this.nodes[i] = temp_nodes_vals[i].doubleValue();
        }
    }



}