package NN;
import java.lang.Math;

public class Tanh implements Activations<Double>{

    //method for the tanh function
    @Override
    public Double activate(Double x){
        double temp = x.doubleValue();
        return(new Double((Math.exp(temp) - Math.exp(-temp)) / (Math.exp(temp) + Math.exp(-temp))));
    }
}