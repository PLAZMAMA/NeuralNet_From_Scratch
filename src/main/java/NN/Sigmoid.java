package NN;

import java.lang.Math;

public class Sigmoid implements Activations<Double>{

    //method for the sigmoid function
    public Double activate(Double x){
        //returns 1 devided by 1 plus the negetive natural base (e) to the power of the negetive of the given value
        return(new Double(1.0 / (1.0 + Math.exp(-x.doubleValue()))));
    }

    //method for the derivative of the sigmoid function
    public Double deriv_activate(Double x){
        //returns the derivative of the sigmoid function
        return(this.activate(x) - (1 - this.activate(x)));
    }
}