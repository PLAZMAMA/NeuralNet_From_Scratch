package NN;

import static org.junit.Assert.assertArrayEquals;

import java.lang.ref.SoftReference;

import javax.annotation.Tainted;

import org.junit.Before;
import org.junit.Test;

import NN.OutputLayer;
import NN.Softmax;

public class OutputLayerTest{
    //instance varivales
    private OutputLayer output;
    
    @Before
    public void init(){
        Softmax soft = new Softmax();
        output = new OutputLayer(soft, 3);
    }

    @Test
    public void test_create_nodes(){
        double[] last_layer_vals = {0.4, 0.9, 0.2, 0.5, 0.8};
        double[][] weights = {{0.1, 0.5, 0.6}, {0.5, 0.9, 0.7}, {-1.2, -1.0, -0.9}, {1.0, -0.9, -0.6}, {0.8, -0.8, 1.3}};
        double[] expected = {0.44862003225352, 0.084451404658485, 0.466928563088};
        double[] actual = new double[3];
        output.calculate_nodes(last_layer_vals, weights);
        for(int i = 0; i < actual.length; i++){
            actual[i] = output.nodes[i].doubleValue();
        }
        assertArrayEquals(expected, actual, 0.00001);
    }
}