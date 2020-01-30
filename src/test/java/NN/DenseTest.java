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
    public void test_constuctor(){
        double expected = 0.0;
        assertEquals("the bias was not initiallized correctly or there is not acess to the bias of dense", expected, this.dense.bias, 0.00001);
    }

    @Test
    public void test_calculate_nodes(){
        this.dense.bias = 0.264;
        double[] last_layer_vals = {0.5, 0.4, 0.7};
        Double[][] weights = {{0.2, -0.6, 1.8, 0.3, 0.9}, {0.1, 1.5, -1.3, 1.2, -1.0}, {1.2, 1.0, -0.4, -0.1, 0.1}};
        this.dense.calculate_nodes(last_layer_vals, weights);
        double[] expected = {0.776259, 0.7797139, 0.5900083, 0.6950847, 0.59483749};
        assertArrayEquals("the calculate_nodes method does not work or there is not access to the nodes of dense", expected, this.dense.nodes, 0.0001);
    }
}