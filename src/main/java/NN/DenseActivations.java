package NN;

/*
* interface for the activation functions that would be used for the hidden / dense layers
*/
interface DenseActivations{ 
    public static double activate(double x){
        System.out.println("has not been overrided from the DenseActivations interface, hence it will return 0.0");
        return(0.0);
    }
}