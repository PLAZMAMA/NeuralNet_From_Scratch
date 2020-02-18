package NN;

interface Cost{
    public double calculate_cost(double[] output, double[] label);
    public double deriv_calculate(double[] output, double[] label);
}