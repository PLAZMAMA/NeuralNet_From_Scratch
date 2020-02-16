package NN;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Before;
import org.junit.Test;

import NN.Model;
import NN.Dense;
import NN.InputLayer;
import NN.OutputLayer;
import NN.Tanh;
import NN.LeakyRelu;
import NN.Softmax;

public class ModelTest{
    //instance variables
    Model model;
    InputLayer input;
    Dense dense1;
    Dense dense2;
    OutputLayer output;

    //removed the creation of the instance of the object model due to some tests not needing it(currently only one)
    @Before
    public void init(){
        double[] data = {4.5, 2.3, 1.9, 9.7, 8.4};
        this.input = new InputLayer(data);
        Tanh tanh = new Tanh();
        this.dense1 = new Dense(tanh, 10);
        LeakyRelu l_relu = new LeakyRelu();
        this.dense2 = new Dense(l_relu, 5);
        Softmax soft = new Softmax();
        this.output = new OutputLayer(soft, 2);
    }

    @Test
    public void test_constructor(){
        this.model = new Model(this.input, this.dense1, this.dense2, this.output);
    }

    @Test
    public void test_create_weights(){
        int[] expected = {5, 3};
        Double[][] output = Model.create_weights(3, 5);
        int[] actual = {output.length, output[0].length};
        assertArrayEquals("the size of the arrays is incorrect", expected, actual);
    }

    //just an integration test due to the function just beign calls to the layers functions
    @Test
    public void test_feed_forward(){
        this.model = new Model(this.input, this.dense1, this.dense2, this.output);
        this.model.feed_forward();
    }
}