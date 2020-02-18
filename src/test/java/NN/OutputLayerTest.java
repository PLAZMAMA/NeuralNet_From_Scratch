package NN;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

import NN.OutputLayer;
import NN.Softmax;
import NN.Sigmoid;

public class OutputLayerTest{
    //instance varivales
    private OutputLayer output;
    
    @Before
    public void init(){
        Softmax soft = new Softmax();
        output = new OutputLayer(soft, 3);
    }

    @Test
    public void test_calculate_nodes(){
        double[] last_layer_vals = {0.4, 0.9, 0.2, 0.5, 0.8};
        Double[][] weights = {{0.1, 0.5, -1.2, 1.0, 0.8}, {0.5, 0.9, -1.0, -0.9, -0.8}, {0.6, 0.7, -0.9, -0.6, 1.3}};
        double[] expected = {0.44862003225352, 0.084451404658485, 0.466928563088};
        this.output.calculate_nodes(last_layer_vals, weights);
        assertArrayEquals("test 1 failed", expected, this.output.nodes, 0.00001);
        Sigmoid sig = new Sigmoid();
        this.output = new OutputLayer(sig, 3);
        expected = new double[] {0.80059224315, 0.43045377606, 0.80690131576};
        this.output.calculate_nodes(last_layer_vals, weights);
        assertArrayEquals("test 2 failed", expected, this.output.nodes, 0.00001);
    }

    @Test
    public void test_type(){
        String expected = "Output";
        assertTrue("type is not correct", this.output.type.equals(expected));
    }
}