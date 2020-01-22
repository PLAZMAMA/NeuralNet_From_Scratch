package NN;
import java.lang.Double;

public class LeakyRelu implements Activations<Double>{
    
    //method for the leaky relu function
    @Override
    public Double activate(Double x){
        if (x < 0.0){
        double temp = x.doubleValue();
        x = new Double(0.01 * temp);
        }
        return(x);
    }
}