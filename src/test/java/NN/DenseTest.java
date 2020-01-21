package NN;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import javax.annotation.Tainted;

import org.junit.Before;
import org.junit.Test;

import NN.Dense;

/*
* test class for the Dense classes methods
*/
public class DenseTest{
    @Before
    public void init(){
        Dense dense = new Dense(5, "relu");
    }

    @Test
    public void test_constuctor(){
        double[] expected = {0,0,0,0,0}
        assertEquals(expected, dense.biases);
    }

    @Test
    public void test_get_layer_nodes(){
        double[] last_layer_vals = {0.5, 0.4, 0.7};
        double[][] weights = {{0.2, -0.6, 1.8}, {0.1, 1.5, -1.3}, {1.2, 1.0, -0.4}, {0.6, -0.9, -1.8}, {1.0, 0.8, -0.6}};
        double[] nodes = dense.get_layer_nodes(last_layer_vals, weights);
        double[] expected = {, , , , };
        assertArrayEquals(expected, nodes, );
    }
}