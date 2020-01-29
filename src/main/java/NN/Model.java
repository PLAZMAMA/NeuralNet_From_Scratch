package NN;

import java.util.Random;
import java.lang.Math;

public class Model{
    //instance variables
    Layers[] layers;
    Random r = new Random();

    Model(Layers... layers){
        this.layers = layers;
    }

    private double[][] create_weights(int last_layer_size, int current_layer_size){
        double[][] weights = new double[current_layer_size][last_layer_size];
        for(int row = 0; row < current_layer_size; row++){
            for(int column = 0; column < last_layer_size; column++){
                weights[row][column] = r.nextGaussian() * Math.sqrt(2.0 / last_layer_size + current_layer_size);
            }
        }
        return(weights);
    }

    public void feed_forward(){

    }

    public void back_propagate(){

    }
}