package NN;

import java.lang.Math;

public class Relu implements Activations<Double>{
    
    //method for the relu function
    public Double Activate(Double x){
        return(new Double(Math.max(0, x.doubleValue())));
    }
}