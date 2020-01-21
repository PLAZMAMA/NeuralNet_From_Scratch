package NN;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import NN.Activations;

/*
* this class tests the Activations class methods
*/
public class ActivationsTest{
    @Before
    public void init(){
        Activations act = new Activations();
    }

    @Test
    public void test_sigmoid(){
        assertEquals(0.9933071491, Activations.sigmoid(5.0), 0.00001);
        assertEquals(0.006692850924, Activations.sigmoid(-5.0), 0.00001);
        assertEquals(0.5, Activations.sigmoid(0.0), 0.00001);
    }

    @Test
    public void test_tanh(){
        assertEquals(0.9999092043, Activations.tanh(5.0), 0.00001);
        assertEquals(-0.9999092043, Activations.tanh(-5.0), 0.00001);
        assertEquals(0.0, Activations.tanh(0.0), 0.00001);
    }

    @Test
    public void test_relu(){
        assertEquals(5.0, Activations.relu(5.0), 0.00001);
        assertEquals(0.0, Activations.relu(-5.0), 0.00001);
        assertEquals(0.0, Activations.relu(0.0), 0.00001);
    }

    @Test
    public void test_leaky_relu(){
        assertEquals(5.0, Activations.leaky_relu(5.0), 0.00001);
        assertEquals(-0.05, Activations.leaky_relu(-5.0), 0.00001);
        assertEquals(0, Activations.leaky_relu(0.0), 0.00001);
    }

    @Test
    public void test_softmax(){
        double[] expected = {0.30918442529062, 0.39656663456166, 0.29424894014771};
        double[] inpt = {0.2668685536, 0.515774693, 0.2173567534};
        assertArrayEquals(expected, Activations.softmax(inpt), 0.00001);
    }


}