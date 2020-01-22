package NN;

public class Softmax implements OutputActivations{

    //method for softmax function
    @Override
    public double[] activate(double[] x){
        double[] exponents = new double[x.length];
        double exponents_sum = 0.0;
        double[] result = new double[x.length];
        for(int i = 0; i < x.length; i++){
            exponents[i] = Math.exp(x[i]);
            exponents_sum += exponents[i];
        }
        System.out.println(exponents_sum);
        System.out.flush();
        for(int i = 0; i < x.length; i++){
            System.out.println(exponents[i] / exponents_sum);
            System.out.flush();
            result[i] = exponents[i] / exponents_sum;
        }
        return(result);
    }
}