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

    //calculates the derivative of the cost function above
    public double deriv_calculate(double output, double label){
        //returning the value of the diffrance of the output to the label times two
        return((output - label) * 2.0);
    }
}