package NN;
import java.lang.Math;

public class Tanh implements Activations<Double>{

    //method for the tanh function
    @Override
    public Double activate(Double x){
        // store x's (given value) double value as a double in temp
        double temp = x.doubleValue();

        /*
        returns the exponenet of temp minus the exponent of negetive temp,
        all divided by the exponent of temp plus the exponent of negetive temp
        and casts it all into a Double(class/wrapper of double)
        */
        return(new Double((Math.exp(temp) - Math.exp(-temp)) / (Math.exp(temp) + Math.exp(-temp))));
    }
}