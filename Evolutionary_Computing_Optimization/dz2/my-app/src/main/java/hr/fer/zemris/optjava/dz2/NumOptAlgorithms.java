package hr.fer.zemris.optjava.dz2;

import java.util.Random;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

public class NumOptAlgorithms {
	
	static double epsilon = 1e-4;
	static double K = 2000;
	
	public NumOptAlgorithms() {
	}
	
	public RealVector gradijentniSpust(IFunction function, int iterNum, RealVector ulaz) {
		int n = 0;
		
		if (ulaz == null) {
			Random random = new Random();

	        double[] randomValues = new double[function.getNumberOfVariables()];
	        for (int i = 0; i < function.getNumberOfVariables(); i++) {
	            randomValues[i] = -10 + (10 - (-10)) * random.nextDouble();
	        }
	        ulaz = new ArrayRealVector(randomValues);
		}

		while (n < iterNum) {
			System.out.println(ulaz.toString());
			
			RealVector gradient = function.getGradient(ulaz, false);
			if (gradient.getNorm() < epsilon) {
				break;
			}
			
			double alfa = search_plus_bisection(function, ulaz);
			ulaz = ulaz.subtract(gradient.mapMultiply(alfa));
			
			n++;
		}
		
		return ulaz;
	}
	
	public static double search_plus_bisection(IFunction function, RealVector x) {
		double lambda_lower = 0;
		double lambda_upper = 0.0001;
		
		RealVector d = function.getGradient(x, true);
		while (function.getGradient(x.add(d.mapMultiply(lambda_upper)), false).dotProduct(d) <= 0) {
			lambda_upper = lambda_upper * 2;
		}

		double lambda = 0;
		int k = 0;
		while (k < K) {
			lambda = ((lambda_lower + lambda_upper) / (double)2);
			
			d = function.getGradient(x, true);
			double dl_value = function.getGradient(x.add(d.mapMultiply(lambda_upper)), false).dotProduct(d);
			
			/*d = function.getGradient(x, true);
			RealVector x_new = x.add(d.mapMultiplyToSelf(lambda));
			RealVector d_x_new = function.getGradient(x_new, false);
			double dl_value = d.dotProduct(d_x_new);*/
		    
		    if (dl_value > epsilon) {
		    	lambda_upper = lambda;
		    } else if (dl_value < -epsilon) {
		        lambda_lower = lambda;
		    } else {
		    	return lambda;
		    }
		    
		    k++;
        }
		
		return lambda;
	}
}
