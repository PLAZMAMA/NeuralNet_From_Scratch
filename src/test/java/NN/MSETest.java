package NN;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

public class MSETest{
    //instance variables
    MSE mse;

    @Before
    public void init(){
        this.mse = new MSE();
    }

    @Test
    public void test_calculate_cost(){
        double[] output = {0.3, 0.7, 0.1};
        double[] label = {0, 1.0, 0};
        double actual = this.mse.calculate_cost(output, label);
        double expected = 0.063333;
        assertEquals(expected, actual, 0.0001);
    }

}