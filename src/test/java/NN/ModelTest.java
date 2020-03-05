package NN;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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
    double[] test_data = {4.5, 2.3, 1.9, 9.7, 8.4};
    Model model;
    InputLayer input;
    Dense dense1;
    Dense dense2;
    OutputLayer output;

    //removed the creation of the instance of the object model due to some tests not needing it(currently only one)
    @Before
    public void init(){
        this.input = new InputLayer(5);
        this.input.nodes = this.test_data;
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

    @Test
    public void test_predict(){
        //initializing the model
        this.model = new Model(this.input, this.dense1, this.dense2, this.output);

        //feed forwarding the data that was given in the @Before to the input layer
        double[] prediction = this.model.predict(this.test_data);

        //testing the prediction
        assertEquals(this.model.layers[3].nodes, prediction);
    }

    @Test
    public void test_train(){
        double[][] x_train = {{5.1,3.5,1.4,0.2}, {5.9,3.0,5.1,1.8}, {6.7,3.0,5.0,1.7}, {5.7,2.6,3.5,1.0}, {7.7,3.0,6.1,2.3}, {4.9,3.0,1.4,0.2}, {6.7,3.3,5.7,2.5}, {4.7,3.2,1.3,0.2}, {5.5,2.5,4.0,1.3}, {4.6,3.1,1.5,0.2}, {6.5,3.0,5.2,2.0}};
        double[][] y_train = {{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}};
        double[] x_test = {6.8,2.8,4.8,1.4};
        double[] y_test = {0.0, 0.0, 1.0};
        
        //changing the model for simplicity and to avoid the vanishing gradient problem
        MSE mse = new MSE();
        Relu relu = new Relu();
        Sigmoid sig = new Sigmoid();
        this.input = new InputLayer(4);
        this.dense1 = new Dense(relu, 8);
        this.dense2 = new Dense(relu, 5);
        this.output = new OutputLayer(sig, 3);
        this.model = new Model(this.input, this.dense1, this.dense2, this.output);

        //getting the initial cost for the test data
        double[] prediction = this.model.predict(x_test);
        double initial_cost = mse.calculate_cost(prediction, y_test);

        //training the model on the data
        this.model.train(x_train, y_train, 2, 2, 0.01);

        //checking if the new cost is smaller than the initial cost
        prediction = this.model.predict(x_test);
        double new_cost = mse.calculate_cost(prediction, y_test);

        assertTrue(new_cost < initial_cost);
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
        //recreating the network to make it easier to calculate and follow
        double learning_rate = 0.01;
        double[] data = {4.5, 2.3};
        this.input = new InputLayer(2);
        this.input.nodes = data;
        LeakyRelu leaky_relu = new LeakyRelu();
        this.dense1 = new Dense(leaky_relu, 3);
        Sigmoid sig = new Sigmoid();
        this.output = new OutputLayer(sig, 2);
        this.model = new Model(this.input, this.dense1, this.dense2, this.output);
        double[] dc_dal = this.calculate_dc_dal();
        double[][] dc_dw;
        double[] dc_db;

        //getting the actual answer
        Model actual_model = new Model(this.input, this.dense1, this.dense2, this.output);

        //copying the model's weights into the actual_model weights
        for(int layer = this.model.layers.length - 1; layer > 0; layer--){
            for(int node = 0; node < this.model.layers[layer].nodes.length; node++){
                for(int weight = 0; weight < this.model.weights[layer][node].length; weight++){
                    actual_model.weights[layer][node][weight] = this.model.weights[layer][node][weight];
                }
            }
        }
        actual_model.back_propagate(dc_dal, learning_rate);

        //feedforwarding both models
        this.model.feed_forward();
        actual_model.feed_forward();

        for(int layer = this.model.layers.length - 1; layer > 0; layer--){
            //getting all the partial derivatives
            dc_dw = this.model.calculate_dc_dws(dc_dal, layer);//gradients of the weights
            dc_db = this.model.calculate_dc_dbs(dc_dal, layer);//gradients of the biases
            if(layer > 1){
                dc_dal = this.model.calculate_dc_dpa(dc_dal, layer);//the derivative of the previouse activation to the cost func
            }

            //updating the weights and biases according to the gradients
            for(int row = 0; row < this.model.weights[layer].length; row++){
                //updating the weights
                for(int column = 0; column < this.model.weights[layer][row].length; column++){
                    this.model.weights[layer][row][column] -= learning_rate * dc_dw[row][column];
                }

                //updating the biases so they won't require a seperate loop
                this.model.layers[layer].biases[row] -= learning_rate * dc_db[row];
            }
        }

        //comparing the results
        for(int layer = this.model.layers.length - 1; layer > 0; layer--){
            for(int node = 0; node < this.model.layers[layer].nodes.length; node++){
                //comparing the weights
                assertArrayEquals("weights test on layer " + layer + " and on node " + node + " failed", this.model.weights[layer][node], actual_model.weights[layer][node], 0.0001);

                //comparing the biases
                assertEquals("biases test on layer " + layer + " and on node " + node + " failed", this.model.layers[layer].biases[node], actual_model.layers[layer].biases[node], 0.0001);
            }

            if(layer > 1){
                assertArrayEquals("previous activation test on layer " + layer, this.model.layers[layer - 1].nodes, actual_model.layers[layer - 1].nodes, 0.0001);
            }
        }
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
        this.model.feed_forward();

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
            z = 0.0;
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
            for(int j = 0; j < expected[i].length; j++){
                //System.out.println(expected[i][j] + ", " + actual[i][j]);
            }
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
                z += this.model.weights[3][column][i] * this.model.layers[2].nodes[i];
            }
            z += this.model.layers[3].biases[column];

            //calculating dc_dpl(pl = previous layer) by adding up each partial derivative with the next one
            for(int row = 0; row < this.model.weights[3][column].length; row++){
                expected[row] += dc_dal[column] * this.model.layers[3].activation.deriv_activate(z) * this.model.weights[3][column][row];
            }
            z = 0.0;
        }

        //testing the expected against the actual
        assertArrayEquals(expected, actual, 0.0001);
    }
}