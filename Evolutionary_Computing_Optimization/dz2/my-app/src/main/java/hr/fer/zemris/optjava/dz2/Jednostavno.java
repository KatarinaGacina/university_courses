
package hr.fer.zemris.optjava.dz2;

import java.util.Random;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

public class Jednostavno {
	
	public static void main(String[] args) {
		if (args.length < 2) {
			throw new IllegalArgumentException("Insufficient number of arguments provided.");
		}
		
		int postupak = Integer.parseInt(args[0]);
		int maxIter = Integer.parseInt(args[1]);
		
		Double x0;
		Double x1;
		if (args.length >= 4) {
			x0 = Double.parseDouble(args[2]);
			x1 = Double.parseDouble(args[3]);
		} else {
			Random rand = new Random();
			
			x0 = rand.nextDouble();
			x1 = rand.nextDouble();
		}
		RealVector x = new ArrayRealVector(new double[]{x0, x1});
		
		NumOptAlgorithms opt = new NumOptAlgorithms();
		
		if (postupak == 1) {
			IFunction f1 = new Function1();
			
			opt.gradijentniSpust(f1, maxIter, x);
			
		} else if (postupak == 2) {
			IFunction f2 = new Function2();
			
			opt.gradijentniSpust(f2, maxIter, x);
			
		}
		
	}
}
