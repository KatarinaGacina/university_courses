package hr.fer.zemris.optjava.dz2;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

public class Function1 implements IFunction {
	
	// f1(x1,x2)=x1**2+(x2−1)**2 

	public int getNumberOfVariables() {
		return 2;
	}

	public double getValue(RealVector x) {
		return Math.pow(x.getEntry(0), 2) + Math.pow((x.getEntry(1) - 1), 2);
	}

	public RealVector getGradient(RealVector x, boolean neg) {
		double dx1 = 2 * x.getEntry(0);
		double dx2 = 2 * (x.getEntry(1) - 1);
		
		if (neg) {
			return new ArrayRealVector(new double[]{-dx1, -dx2});
		} else {
			return new ArrayRealVector(new double[]{dx1, dx2});
		}
	}

}
