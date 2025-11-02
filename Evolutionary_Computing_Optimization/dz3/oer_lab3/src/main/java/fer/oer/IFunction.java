package fer.oer;

import org.apache.commons.math3.linear.RealVector;

public interface IFunction {
	public int getNumberOfVariables();
	
	public double getValue(RealVector x);
}
