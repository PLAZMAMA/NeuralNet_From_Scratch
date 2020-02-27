package NN;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Before;
import org.junit.Test;
import java.util.Arrays;
import java.util.Random;

import NN.Model;
import NN.Dense;
import NN.InputLayer;
import NN.OutputLayer;
import NN.Tanh;
import NN.LeakyRelu;
import NN.Softmax;
import NN.Sigmoid;
import NN.MSE;

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
        double[][] output = Model.create_weights(3, 5);
        int[] actual = {output.length, output[0].length};
        assertArrayEquals("the size of the arrays is incorrect", expected, actual);
    }

    //just an integration test due to the function just beign calls to the layers functions
    @Test
    public void test_feed_forward(){
        this.model = new Model(this.input, this.dense1, this.dense2, this.output);
        this.model.feed_forward();
    }

    //test for the back_propagation method(currently will only test with an outputlayer of a basic activation function(aka not softmax))
    @Test
    public void test_back_propagate(){
        //recreating the network to make it easier to calculate by hand
        double[] data = {4.5, 2.3};
        this.input = new InputLayer(data);
        LeakyRelu leaky_relu = new LeakyRelu();
        this.dense1 = new Dense(leaky_relu, 3);
        Sigmoid sig = new Sigmoid();
        this.output = new OutputLayer(sig, 2);
        this.model = new Model(this.input, this.dense1, this.dense2, this.output);
    }

    /*
    method for calculating the partial derivative of al with respect to the cost function for a random label, changing
    the output layer activation into a sigmoid from softmax and the number of nodes
    (used in test_calculate_dc_dws, test_calculate_dc_dbs and calculate_dc_pa)
    */
    private double[] calculate_dc_dal(){
        Random r = new Random();
        MSE mse = new MSE();
        Sigmoid sig = new Sigmoid();
        this.output = new OutputLayer(sig, 2);
        this.model = new Model(this.input, this.dense1, this.dense2, this.output);

        //creating a random label to test against
        double[] label = new double[this.output.nodes.length];
        Arrays.fill(label, 0.0);
        label[r.nextInt(label.length)] = 1.0; //chosing a random label to become 1.0 like a one-hot array

        //calculating the partial derivative of al with respect to the cost function
        double[] dc_dal = new double[this.output.nodes.length];
        for(int i = 0; i < this.output.nodes.length; i++){
            dc_dal[i] = mse.deriv_calculate(this.output.nodes[i], label[i]);
        }
        return(dc_dal);
    }

    @Test
    public void test_calculate_dc_dws(){
        Sigmoid sig = new Sigmoid(); //initialzing sigmoid due to the derivative being used later

        //getting the partial derivative of al with respect to the cost function for a random label
        double[] dc_dal = this.calculate_dc_dal();

        //creating the expected and checking if it equals to the actual
        double[][] actual = this.model.calculate_dc_dws(dc_dal, 3);
        double[][] expected = new double[this.output.nodes.length][this.dense2.nodes.length];
        double z;
        for(int row = 0; row < expected.length; row++){

            //geting the z
            z = 0;
            for(int i = 0; i < this.model.layers[2].nodes.length; i++){
                z += this.model.weights[3][row][i] * this.model.layers[2].nodes[i];
            }
            z += this.model.layers[3].biases[row];

            //getting the expected value
            for(int column = 0; column < expected[row].length; column++){
                expected[row][column] = dc_dal[row] * sig.deriv_activate(z) * this.model.layers[2].nodes[column];
            }
        }
        for(int i = 0; i < expected.length; i++){
            assertArrayEquals("test " + i + " failed", expected[i], actual[i], 0.0001);
        }
    }
    
    @Test
    public void test_calculate_dc_dbs(){
        Sigmoid sig = new Sigmoid(); //initialzing sigmoid due to the derivative being used later
        
        //getting the partial derivative of the outputlayer activation with respect to the cost function of a random label
        double[] dc_dal = this.calculate_dc_dal();
        
        //creating the expected and checking if it equals to the actual
        double[] actual = this.model.calculate_dc_dbs(dc_dal, 3);
        double[] expected = new double[this.output.nodes.length];
        double z;

        //getting the expected and the z
        for(int node = 0; node < this.model.layers[3].biases.length; node++){
            //calculating the z
            z = 0.0;
            for(int column = 0; column < this.model.weights[3][node].length; column++){
                z += this.model.weights[3][node][column] * this.model.layers[2].nodes[column];
            }
            z += this.model.layers[3].biases[node];

            //calculating the expected
            expected[node] = dc_dal[node] * sig.deriv_activate(z);
        }

        assertArrayEquals(expected, actual, 0.0001);
    }

    @Test
    public void test_calculate_dc_pa(){
        //getting the partial derivative of the outputlayer activation with respect to the cost function of a random label
        double[] dc_dal = this.calculate_dc_dal();
        double z = 0;
        double[] actual = this.model.calculate_dc_dpa(dc_dal, 3);
        double[] expected = new double[this.dense2.nodes.length];
        Arrays.fill(expected, 0.0); //filling the array with zeros since I will have to get the sum of the partial derivatives
        for(int column = 0; column < this.model.weights[3].length; column++){
            //calculating z
            for(int i = 0; i < this.model.weights[3][column].length; i++){
                z += this.model.weights[3][column][i] * this.model.layers[3].nodes[column];
            }
            z += this.model.layers[3].biases[column];

            //calculating dc_dpl(pl = previous layer) by adding up each partial derivative with the next one
            for(int row = 0; row < this.model.weights[3][column].length; row++){
                expected[row] += dc_dal[column] * this.model.layers[3].activation.deriv_activate(z) * this.model.weights[3][column][row];
            }
        }

        //testing the expected against the actual
        assertArrayEquals(expected, actual, 0.0001);
    }
}