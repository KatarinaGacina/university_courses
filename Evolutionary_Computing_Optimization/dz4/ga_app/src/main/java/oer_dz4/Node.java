package oer_dz4;

import java.util.List;

public abstract class Node {
	
	public abstract double evaluate(List<Double> X);
	
	public abstract int treeDepth();
	
	public abstract int countNodes();
	
	public abstract String printExpression();
	
	public abstract Node copyTree(Node p);
	
	public abstract Node getByIndex(int i);

	public abstract Node getParent();

	public abstract void setParent(Node parent);

}
