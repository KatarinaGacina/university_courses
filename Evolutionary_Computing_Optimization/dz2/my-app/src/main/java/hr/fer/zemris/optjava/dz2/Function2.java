package hr.fer.zemris.optjava.dz2;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

public class Function2 implements IFunction {

	// f2(x1,x2)=(x1−1)**2+10⋅(x2−2)**2 
	
	public int getNumberOfVariables() {
		return 2;
	}

	public double getValue(RealVector x) {
		return Math.pow((x.getEntry(0) - 1), 2) + 10 * Math.pow((x.getEntry(1) - 2), 2);
	}

	public RealVector getGradient(RealVector x, boolean neg) {
		double dx1 = 2 * (x.getEntry(0) - 1);
		double dx2 = 20 * (x.getEntry(1) - 2);
				
		if (neg) {
			return new ArrayRealVector(new double[]{-dx1, -dx2});
		} else {
			return new ArrayRealVector(new double[]{dx1, dx2});
		}
	}
}
