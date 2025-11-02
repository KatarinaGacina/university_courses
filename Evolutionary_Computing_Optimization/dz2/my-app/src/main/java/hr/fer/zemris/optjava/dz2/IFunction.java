package hr.fer.zemris.optjava.dz2;

import org.apache.commons.math3.linear.RealVector;

public interface IFunction {
	
	public int getNumberOfVariables();
	
	public double getValue(RealVector x);
	
	public RealVector getGradient(RealVector x, boolean neg);
	
}
