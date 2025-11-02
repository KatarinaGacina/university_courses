package fer.oer;

import java.util.Arrays;

import org.apache.commons.math3.linear.ArrayRealVector;

public class RegresijaSustava {
	
    public static void main(String[] args) {
    	
    	String datoteka = args[0];
    	IFunction f4 = new Prijenosna(datoteka);
    	
    	SimulatedAnnealing sk = new SimulatedAnnealing(false, f4);
    	
    	double[] rjesenje = sk.startAlgorithm();
    	System.out.println();
    	System.out.println("Rjesenje:");
    	System.out.println(Arrays.toString(rjesenje));
    	System.out.println("Pogreska:");
    	System.out.println(f4.getValue(new ArrayRealVector(rjesenje)));
    }	
}
