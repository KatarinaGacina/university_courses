package oer_dz4;

import java.util.List;

public class ConstLeaf extends Node {
	
	private double value;
	private Node parent;
	
	public ConstLeaf(Double n, Node parent) {
        this.value = n;
        this.setParent(parent);
    }
	
	Double getValue() {
		return value;
	}

	@Override
	public double evaluate(List<Double> X) {
		return value;
	}

	@Override
	public int treeDepth() {
		return 1;
	}

	@Override
	public int countNodes() {
		return 1;
	}

	@Override
	public String printExpression() {
		return String.valueOf(value);
	}
	
	public Node copyTree(Node p) {
		ConstLeaf n = new ConstLeaf(value, p);
		return n;
	}

	@Override
	public Node getByIndex(int i) {
		return this;
	}

	public Node getParent() {
		return parent;
	}

	public void setParent(Node parent) {
		this.parent = parent;
	}
}
