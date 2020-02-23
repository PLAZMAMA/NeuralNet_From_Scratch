package NN;

import java.util.HashMap;
import java.util.Random;
import java.lang.Math;

import NN.MSE;


public class Model{
    //instance variables
    Layers[] layers;
    double[][][] weights;
    MSE cost;

    Model(Layers... layers){
        this.layers = layers;
        this.weights = new double[this.layers.length][][];
        this.cost = new MSE();
        double[][] current_layer_weights;

        /*
        initializing all the weights
        the hashmap of the weights will look like so:
        {
        layer_(number of the layer, the first aka input is 0, the second is 1 and so on):
        the weights that connect the layer before the corresponding to the corresponding layer, ...
        }
        */
        for(int i = 1; i < this.layers.length; i++){
            current_layer_weights = create_weights(this.layers[i-1].nodes.length, this.layers[i].nodes.length);
            this.weights[i] = current_layer_weights;
        }
        
    }

    /*
    *method that creates random weights given the last layer's size and the current one's.
    *the random assignment of weights is done by getting a random value from a standart distribution
    *and multiply it by the square root of 2 divided by the last layer size plus the current layer size
    *to try to reduce the vanishing and exploding gradient problems.
    */
    public static double[][] create_weights(int last_layer_size, int current_layer_size){
        Random r = new Random();
        double[][] weights = new double[current_layer_size][last_layer_size];
        for(int row = 0; row < current_layer_size; row++){
            for(int column = 0; column < last_layer_size; column++){
                weights[row][column] = r.nextGaussian() * Math.sqrt(2.0 / last_layer_size + current_layer_size);
            }
        }
        return(weights);
    }

    //calculates each layer's (beside the input layer since it just the data technically) nodes in sequetial order from the first (not counting the input) to the output
    public void feed_forward(){
        for(int i = 1; i < this.layers.length; i++){
            layers[i].calculate_nodes(this.layers[i-1].nodes, weights[i]);
        }
    }

    //this method will change the weights and biases based on the loss of the cost function
    public void back_propagate(double output, double label){
        
    }

    //calculates the partial derivative of the weights of a given layer with respet to the cost
    public double[][] calculate_dc_dws(double[] dc_dal, int layer){
        double[][] dc_dws = new double[this.weights[layer].length][this.weights[layer][0].length];
        double z; //the weight * the last node value + the bias
        
        //calculating the z and using the chain rule to figure out each dc_dw of the given layer
        for(int row = 0; row < dc_dws.length; row++){
            for(int column = 0; column < dc_dws[row].length; column++){
                z = this.weights[layer][row][column] * this.layers[layer - 1].nodes[column] + this.layers[layer - 1].biases[column];
                dc_dws[row][column] = dc_dal[row] * this.layers[layer].activation.activate(z) * this.layers[layer - 1].nodes[column];
            }
        }

        return(dc_dws);
    }
}