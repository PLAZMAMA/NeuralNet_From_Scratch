package NN;

interface Cost{
    public abstract double calculate_cost(double[] output, double[] label);
}