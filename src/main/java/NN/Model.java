package NN;

import java.util.Random;

import java.lang.Math;
import java.util.Arrays;

import NN.MSE;


public class Model{
    //instance variables
    Layers[] layers;
    double[][][] weights;
    MSE cost;

    Model(Layers... layers){
        this.layers = layers;
        this.weights = new double[this.layers.length][][];
        this.cost = new MSE();
        double[][] current_layer_weights;

        /*
        initializing all the weights
        the hashmap of the weights will look like so:
        {
        layer_(number of the layer, the first aka input is 0, the second is 1 and so on):
        the weights that connect the layer before the corresponding to the corresponding layer, ...
        }
        */
        for(int i = 1; i < this.layers.length; i++){
            current_layer_weights = create_weights(this.layers[i-1].nodes.length, this.layers[i].nodes.length);
            this.weights[i] = current_layer_weights;
        }
        
    }

    /*
    *method that creates random weights given the last layer's size and the current one's.
    *the random assignment of weights is done by getting a random value from a standart distribution
    *and multiply it by the square root of 2 divided by the last layer size plus the current layer size
    *to try to reduce the vanishing and exploding gradient problems.
    */
    public static double[][] create_weights(int last_layer_size, int current_layer_size){
        Random r = new Random();
        double[][] weights = new double[current_layer_size][last_layer_size];
        for(int row = 0; row < current_layer_size; row++){
            for(int column = 0; column < last_layer_size; column++){
                weights[row][column] = r.nextGaussian() * Math.sqrt(2.0 / last_layer_size + current_layer_size);
            }
        }
        return(weights);
    }

    public double[] predict(double[] x){
        //checking if the data that is given is equal in length to the input layer(if the length of x matches the input shape)
        if(layers[0].nodes.length != x.length){
            System.out.println("input shape miss matched: expected an array of size " + this.layers[0].nodes.length + " but got " + x.length);
        }
        this.layers[0].nodes = x;
        this.feed_forward();
        return(this.layers[this.layers.length - 1].nodes);
    }

    //this method trains the neural network(aka model)
    public void train(double[][] x, double[][] y, int batch_size, int epochs, double learning_rate){
        double[] output = new double[x.length];
        double[] average = new double[x.length];
        int batches = (int) Math.ceil(x.length / (double) batch_size);
        int batch_indx;
        //itterates over each epoch
        for(int epoch = 0; epoch < epochs; epoch++){
            batch_indx = 0;
            //itterates over each batch
            for(int batch = 0; batch < batches; batch++){
                Arrays.fill(average, 0.0);

                while(batch_indx < batch_indx + batch_size && batch_indx < x.length){
                    output = this.predict(x[batch_indx]);
                    //takes the output, get dc_dal of that output and adds it to the average
                    for(int al = 0; al < y[0].length; al++){
                        average[al] += this.cost.deriv_calculate(output[al], y[batch_indx][al]);
                    }
                    batch_indx++;
                }

                //averages the sum of by dividing each output by the batch size
                for(int i = 0; i < average.length; i++){
                }
                

                //backpropagating the average dc_dal
                this.back_propagate(average, learning_rate);
            }

            //printing the cost of the last training input and epoch progress
            System.out.println("Epoch " + (epoch + 1) + "/" + epochs + "    loss: " + this.cost.calculate_cost(this.layers[layers.length - 1].nodes, y[y.length - 1]));
        }
    }



    //calculates each layer's (beside the input layer since it just the data technically) nodes in sequetial order from the first (not counting the input) to the output
    public void feed_forward(){
        for(int i = 1; i < this.layers.length; i++){
            layers[i].calculate_nodes(this.layers[i-1].nodes, weights[i]);
        }
    }

    //this method will change the weights and biases based on the loss of the cost function
    public void back_propagate(double[] dc_dal, double learning_rate){
        double[][] dc_dws;
        double[] dc_dbs;
        for(int layer = this.layers.length - 1; layer > 0; layer--){
            //calculating the partial derivative of the weights of the layer with respect to the cost
            dc_dws = this.calculate_dc_dws(dc_dal, layer);

            //calculating the partial derivative of the biases of the layer with respect to the cost
            dc_dbs = this.calculate_dc_dbs(dc_dal, layer);

            //calculating the partial derivative of the previous activation of the layer with respect to the cost if its not the first dense layers
            if(layer > 1){
                dc_dal = this.calculate_dc_dpa(dc_dal, layer);
            }

            //updating the weights and biases with the given learning rate
            for(int node = 0; node < this.layers[layer].nodes.length; node++){
                //updating the weights
                for(int weight = 0; weight < this.weights[layer][node].length; weight++){
                    this.weights[layer][node][weight] -= learning_rate * dc_dws[node][weight];
                }

                //updating the biases
                this.layers[layer].biases[node] -= learning_rate * dc_dbs[node];
            }
        }
    }

    //calculates the partial derivative of the weights of a given layer with respet to the cost
    public double[][] calculate_dc_dws(double[] dc_dal, int layer){
        double[][] dc_dws = new double[this.weights[layer].length][this.weights[layer][0].length];
        double z; //the weight * the last node value + the bias
        
        //calculating the z and using the chain rule to figure out each dc_dw of the given layer
        for(int row = 0; row < dc_dws.length; row++){
            //calculating z
            z = 0.0;
            for(int i = 0; i < this.weights[layer][row].length; i++){
                z += this.weights[layer][row][i] * this.layers[layer - 1].nodes[i];
            }
            z += this.layers[layer].biases[row];

            //calculating dc_dws
            for(int column = 0; column < dc_dws[row].length; column++){
                dc_dws[row][column] = dc_dal[row] * this.layers[layer].activation.deriv_activate(z) * this.layers[layer - 1].nodes[column];
            }
        }

        return(dc_dws);
    }

    //calculates the partial derivative of the biases with respect to the cost function
    public double[] calculate_dc_dbs(double[] dc_dal, int layer){
        double[] dc_dbs = new double[dc_dal.length];
        double z;
        //calculating z and the partial derivatives of dc_db
        for(int node = 0; node < this.layers[layer].nodes.length; node++){
            //calculating z
            z = 0.0;
            for(int i = 0; i < this.layers[layer - 1].nodes.length; i++){
                z += this.weights[layer][node][i] * this.layers[layer - 1].nodes[i];
            }
            z += this.layers[layer].biases[node];

            dc_dbs[node] = dc_dal[node] * this.layers[layer].activation.deriv_activate(z);
        }
        return(dc_dbs);
    }

    //calculates the partial derivative of the previous activations with respect to the cost function given 
    public double[] calculate_dc_dpa(double[] dc_dal, int layer){
        double z = 0.0;
        double[] dc_dpa = new double[this.layers[layer - 1].nodes.length];
        Arrays.fill(dc_dpa, 0.0);
        /*
        looping on each of the nodes of the given layer (due to the partial derivative of the
        previous activation is the sum of each partial derivative of the activation with respect to the cost)
        */
        for(int node = 0; node < this.layers[layer].nodes.length; node++){
            //calculating z
            for(int i = 0; i < this.weights[layer][node].length; i++){
                z += this.weights[layer][node][i] * this.layers[layer - 1].nodes[i];
            }
            z += this.layers[layer].biases[node];

            //calculating the partial derivative of the previous activation throught the current node
            for(int pa = 0; pa < this.weights[layer][node].length; pa++){
                dc_dpa[pa] += dc_dal[node] * this.layers[layer].activation.deriv_activate(z) * this.weights[layer][node][pa];
            }
            z = 0.0;
        }

        return(dc_dpa);
    }
}