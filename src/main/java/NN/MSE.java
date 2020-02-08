package NN;

import java.lang.Math;

public class MSE implements Cost{
    
    //calculates the mean square error given the output and label(label must be a one hot array)
    public double calculate_cost(double[] output, double[] label){
        //getting the sum of the diffrence squared((output - label) ^ 2)
        double sum = 0;
        for(int i = 0; i < output.length; i++){
            sum += Math.pow(output[i] - label[i], 2);
        }

        //returning the average of the squard diffrence values by dividing the sum by the number of numbers in label or output
        return(sum / output.length);
    }
}