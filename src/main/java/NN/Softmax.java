package NN;
import java.lang.Math;

public class Softmax implements Activations<Double[]>{

    //method for softmax function
    @Override
    public Double[] activate(Double[] x){
        double[] exponents = new double[x.length];
        double exponents_sum = 0.0;
        Double[] result = new Double[x.length];
        for(int i = 0; i < x.length; i++){
            exponents[i] = Math.exp(x[i].doubleValue());
            exponents_sum += exponents[i];
        }
        for(int i = 0; i < x.length; i++){
            result[i] = new Double(exponents[i] / exponents_sum);
        }
        return(result);
    }
}