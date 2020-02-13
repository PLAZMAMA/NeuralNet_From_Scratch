package NN;

interface Cost{
    public double calculate_cost(double[] output, double[] label);
}