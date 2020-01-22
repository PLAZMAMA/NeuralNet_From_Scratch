package NN;

interface OutputActivations{
    static double[] activate(double[] x){
        System.out.println("has not been overrided from the OutputActivaions interface, hence it will return {0.0}");
        double[] def = {0.0};
        return(def);
    }
}