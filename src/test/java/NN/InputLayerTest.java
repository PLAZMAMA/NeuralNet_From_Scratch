package NN;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

import NN.InputLayer;

public class InputLayerTest{
    //instance varibles
    InputLayer input;
    
    @Before
    public void init(){
        double[] inpt = {1.0, 3.0, 0.8, 0.3, 0.6};
        this.input = new InputLayer(5);
        this.input.nodes = inpt;
    }

    @Test
    public void test_type(){
        String expected = "Input";
        assertTrue("type is not correct", this.input.type.equals(expected));
    }

    @Test
    public void test_constructor(){
        double[] expected = {1.0, 3.0, 0.8, 0.3, 0.6};
        assertArrayEquals("the initialization of the nodes in the constructor is incorrect", expected, this.input.nodes, 0.00001);
    }

}