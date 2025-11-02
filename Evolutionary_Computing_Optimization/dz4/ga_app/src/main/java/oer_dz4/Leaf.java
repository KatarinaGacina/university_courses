package oer_dz4;

import java.util.List;

public class Leaf extends Node {
	
	private int index;
	private Node parent;
	
	public Leaf(int i, Node parent) {
        this.index = i;
        this.setParent(parent);
    }
	
	int getIndex() {
		return index;
	}

	@Override
	public double evaluate(List<Double> X) {
		return X.get(index);
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
		return "x" + String.valueOf(index);
	}

	@Override
	public Node copyTree(Node p) {
		Leaf n = new Leaf(index, p);
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
