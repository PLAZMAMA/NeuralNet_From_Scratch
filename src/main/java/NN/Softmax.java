package NN;
import java.lang.Math;

public class Softmax implements Activations<Double[]>{

    //method for softmax function
    public Double[] activate(Double[] x){
        //initializes all the variables
        double[] exponents = new double[x.length];
        double exponents_sum = 0.0;
        Double[] result = new Double[x.length];

        //gets each values exponent, stores it in an array and adds it to the exponent sum
        for(int i = 0; i < x.length; i++){
            exponents[i] = Math.exp(x[i].doubleValue());
            exponents_sum += exponents[i];
        }

        //iterates over each values exponents, devides it by the exponents sum and then stores it in result
        for(int i = 0; i < x.length; i++){
            result[i] = new Double(exponents[i] / exponents_sum);
        }
        return(result);
    }

    //method for the derivative of the softmax function
    public Double[] deriv_activate(Double[] x){
        //not completed yet
    }
}