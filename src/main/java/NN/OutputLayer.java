package NN;

import NN.Layers;
import java.lang.Double;

//class for the output layer of the neural network(only categorical with softmax for simplicity)
public class OutputLayer extends Layers{
    //instance variables
    Softmax softmax_activation;

    //constructor method for the outputlayer class if a softmax activation is inputed
    OutputLayer(Softmax activation, int n_nodes){
        this.softmax_activation = activation;
        this.init_type_nodes(n_nodes);

        //initialzing the biases
        super.biases = new double[n_nodes];
        for(int i = 0; i < n_nodes; i++){
            super.biases[i] = 0.0;
        }
    }

    //constructor metod for the outputlayer class if a other activation is inputed
    OutputLayer(Activations<Double> activation, int n_nodes){
        super.activation = activation;
        this.init_type_nodes(n_nodes);

        //initialzing the biases
        super.biases = new double[n_nodes];
        for(int i = 0; i < n_nodes; i++){
            super.biases[i] = 1.0;
        }
    }

    /*initializes the nodes with the given nodes size and sets the type of this layer which is used in the model class
    (created this method to avoid rewriting the same code)
    */
    private void init_type_nodes(int n_nodes){
        super.nodes = new double[n_nodes];
        super.type = "Output";
    }

    //calculates the nodes of the output layer. Will be used if softmax was passed as the activation function
    private void calculate_softmax_nodes(double[] last_layer_vals, double[][] weights){
        double sum;
        Double[] temp_nodes_vals = new Double[this.nodes.length];
        for(int row = 0; row < weights.length; row++){

            //gets the sum of the last layer's nodes vals times the corresponding/connected weights
            sum = 0.0;
            for(int column = 0; column < weights[row].length; column++){
                sum += last_layer_vals[column] * weights[row][column];
            }

            //stores the sum of each node in a temporary Double array
            temp_nodes_vals[row] = new Double(sum);
        }

        /*
        the temporary Double array is put into a activation function(in this case probably softmax),
        then the output of the activation function is stored in this.nodes/nodes by dumping the temp array into the 
        */
        temp_nodes_vals = this.softmax_activation.activate(temp_nodes_vals);
        for(int i = 0; i < temp_nodes_vals.length; i++){
            super.nodes[i] = temp_nodes_vals[i].doubleValue();
        }
    }

    //calculates the nodes of the output layer. It will be used if any other activation besides softmax will be used
    private void calculate_activation_nodes(double[] last_layer_vals, double[][] weights){
        double sum = 0.0;

        //iterates over each nodes of the current layer
        for(int row = 0; row < weights.length; row++){

            //gets the sum of the weights times the last layer nodes
            for(int column = 0; column < weights[row].length; column++){
                sum += weights[row][column] * last_layer_vals[column];
            }
            //puts the sum throught the activation, dumps it to super.nodes(nodes of the layer) and resets the sum
            super.nodes[row] = super.activation.activate(sum + super.biases[row]);
            sum = 0.0;
        }
    }

    //calculates the nodes of the output layer
    public void calculate_nodes(double[] last_layer_vals, double[][] weights){
        /*
        checks if the this.activtion is not empty(meaning that the constructor calculate_activation_nodes was used),
        and calls the appropriate method
        */
        if(this.activation != null){
            this.calculate_activation_nodes(last_layer_vals, weights);
        }else{
            this.calculate_softmax_nodes(last_layer_vals, weights);
        }
    }



}