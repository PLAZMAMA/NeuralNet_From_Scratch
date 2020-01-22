package NN;

public class LeakyRelu implements DenseActivations{

    
    //method for the leaky relu function
    @Override
    public double activate(double x){
        if (x < 0.0){
            return(0.01 * x);
        }else{
            return(x);
        }
    }
}