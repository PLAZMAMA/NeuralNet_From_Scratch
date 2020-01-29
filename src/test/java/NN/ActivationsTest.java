package NN;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

import NN.Activations;


//this class tests the all the classes that implement the Activations interface
public class ActivationsTest{

    @Test
    public void test_sigmoid(){
        Activations<Double> act = (Activations<Double>)new Sigmoid();
        assertEquals("test 1 of failed", 0.9933071491, act.activate(5.0), 0.00001);
        assertEquals("test 2 of failed", 0.006692850924, act.activate(-5.0), 0.00001);
        assertEquals("test 3 of failed", 0.5, act.activate(0.0), 0.00001);
    }

    @Test
    public void test_tanh(){
        Activations<Double> act = (Activations<Double>)new Tanh();
        assertEquals("test 1 failed", 0.9999092043, act.activate(5.0), 0.00001);
        assertEquals("test 2 failed", -0.9999092043, act.activate(-5.0), 0.00001);
        assertEquals("test 3 of failed", 0.0, act.activate(0.0), 0.00001);
    }

    @Test
    public void test_relu(){
        Activations<Double> act = (Activations<Double>)new Relu();
        assertEquals("test 1 failed", 5.0, act.activate(5.0), 0.00001);
        assertEquals("test 2 failed", 0.0, act.activate(-5.0), 0.00001);
        assertEquals("test 3 failed", 0.0, act.activate(0.0), 0.00001);
    }

    @Test
    public void test_leaky_relu(){
        Activations<Double> act = (Activations<Double>)new LeakyRelu();
        assertEquals("test 1 failed", 5.0, act.activate(5.0), 0.00001);
        assertEquals("test 2 failed", -0.05, act.activate(-5.0), 0.00001);
        assertEquals("test 3 failed", 0, act.activate(0.0), 0.00001);
    }

    @Test
    public void test_softmax(){
        Activations<Double[]> act = (Activations<Double[]>)new Softmax();
        double[] expected = {0.30918442529062, 0.39656663456166, 0.29424894014771};
        Double[] inpt = {0.2668685536, 0.515774693, 0.2173567534};
        Double[] temp = act.activate(inpt);
        double[] actual = new double[temp.length];
        for(int i = 0; i < temp.length; i++){
            actual[i] = temp[i].doubleValue();
        }

        assertArrayEquals("the softmax function does not work as intended", expected, actual, 0.00001);
    }


}