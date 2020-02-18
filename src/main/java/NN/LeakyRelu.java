package NN;
import java.lang.Double;

public class LeakyRelu implements Activations<Double>{
    
    //method for the leaky relu function
    @Override
    public Double activate(Double x){
        /*
        returns x if x (given value) is bigger than 0, 
        otherwise it returns the x (given value) times 0.01 then all casted to a Double (class/wrapper of double)
        */
        if (x < 0.0){
        x = new Double(0.01 * x);
        }
        return(x);
    }
    
    //method for the derivative of the leaky relu function
    public Double deriv_activate(Double x){
        /*
        if x is bigger than 0 then it returns 1, if x is smaller than 0 it returns 0.01.
        However if x is 0 it is not clear what should be return so I decided on returning a half(0.5)
        */
        if(x > 0.0){
            return(1.0);
        }
        return(0.01);

    }
}