package NN;
import java.lang.Math;
 
/*
* the Activations class has activation functions used for the neural network layers
* current activaion functions: sigmoid, relu, leaky relu, tanh, softmax
*/
public class Activations{
 
   /*
   * method for the sigmoid function
   */
   public static double sigmoid(double x){
       return(1.0 / (1.0 + Math.exp(-x)));
   }
 
   /*
   * method for the tanh function
   */
   public static double tanh(double x){
       return((Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)));
   }
 
   /*
   * method for the relu function
   */
   public static double relu(double x){
       return(Math.max(0, x));
   }
 
   /*
   * method for the leaky relu function
   */
   public static double leaky_relu(double x){
       if (x < 0.0){
           return(0.01 * x);
       }else{
           return(x);
       }
   }
 
   /*
   * method for softmax function
   */
   public static double[] softmax(double[] x){
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
