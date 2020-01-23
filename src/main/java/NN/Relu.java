package NN;

import java.lang.Math;

public class Relu implements Activations<Double>{
    
    //method for the relu function
    public Double activate(Double x){
        //if x(given value) is less than 0, then the function will return 0. otherwise it will return x(given value)
        return(new Double(Math.max(0, x.doubleValue())));
    }
}