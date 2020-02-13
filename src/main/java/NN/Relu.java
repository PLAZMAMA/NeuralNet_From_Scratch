package NN;

import java.lang.Math;

public class Relu implements Activations<Double>{
    
    //method for the relu function
    public Double activate(Double x){
        //if x(given value) is less than it returns 0, then the function will return 0. otherwise it will return x(given value)
        return(new Double(Math.max(0, x.doubleValue())));
    }

    //method for the derivative of the relu function
    public Double deriv_activate(Double x){
        /*
        if x is bigger than 0 then it returns 1, if x is smaller than 0 it returns 0.
        However if x is 0 it is not clear what should be return so I decided on returning a half(0.5)
        */
        if(x > 0.0){
            return(1.0);
        }else if(x < 0.0){
            return(0.0);
        }else{
            return(0.5);
        }
    }
}