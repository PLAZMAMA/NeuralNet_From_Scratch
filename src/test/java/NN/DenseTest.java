package NN;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import NN.Dense;
import NN.Sigmoid;


//test class for the Dense classes methods

public class DenseTest{
    //instance variables
    private Dense dense;

    @Before
    public void init(){
        Sigmoid sig = new Sigmoid();
        this.dense = new Dense(sig, 5);
    }

    @Test
    public void test_bias_initialization(){
        double[] expected = {0.0, 0.0, 0.0, 0.0, 0.0};
        assertArrayEquals("the bias was not initiallized correctly or there is not acess to the bias of dense", expected, this.dense.biases, 0.00001);
    }

    @Test
    public void test_calculate_nodes(){
        double[] new_biases = {0.24, 0.13, 0.64, 0.93, 0.78};
        for(int i = 0; i < this.dense.biases.length; i++){
            this.dense.biases[i] = new_biases[i];
        }
        double[] last_layer_vals = {0.5, 0.4, 0.7};
        Double[][] weights = {{0.2, 0.1, 1.2}, {-0.6, 1.5, 1.0}, {1.8, -1.3, -0.4}, {0.3, 1.2, -0.1}, {0.9, -1.0, 0.1}};
        this.dense.calculate_nodes(last_layer_vals, weights);
        double[] expected = {0.77206354942678, 0.755838899094, 0.67699585623, 0.81607827258, 0.7109495026250}; //these values havn't been put throught the activation function
        assertArrayEquals("the calculate_nodes method does not work or there is not access to the nodes of dense", expected, this.dense.nodes, 0.0001);
    }
}