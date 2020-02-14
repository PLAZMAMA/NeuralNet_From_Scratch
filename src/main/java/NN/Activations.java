package NN;

/*
* interface for the activations classes
*/
interface Activations <T>{
    public T activate(T x);
    public T deriv_activate(T x);
}