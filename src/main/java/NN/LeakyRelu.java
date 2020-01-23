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
        double temp = x.doubleValue();
        x = new Double(0.01 * temp);
        }
        return(x);
    }
}