package NN;

import java.lang.Math;

public class Sigmoid implements Activations<Double>{

    //method for the sigmoid function
    public Double activate(Double x){
        return(new Double(1.0 / (1.0 + Math.exp(-x.doubleValue()))));
    }
}