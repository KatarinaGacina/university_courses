package oer_dz4;

import java.util.List;

public class BinaryNode extends Node {
	
	private IBinaryOperator binaryOperator;
	
	private Node parent;
	private Node leftChild;
	private Node rightChild;
    
    public BinaryNode(IBinaryOperator o, Node parent) {
        this.binaryOperator = o;
        this.parent = parent;
        
        leftChild = rightChild = null;
    }

	IBinaryOperator getOperator() {
		return binaryOperator;
	}
	
	protected Node getLeftChild() {
		return leftChild;
	}

	protected Node getRightChild() {
		return rightChild;
	}

	protected void setLeftChild(Node leftChild) {
		this.leftChild = leftChild;
	}

	protected void setRightChild(Node rightChild) {
		this.rightChild = rightChild;
	}

	@Override
	public double evaluate(List<Double> X) {
		return binaryOperator.result(this.leftChild.evaluate(X), this.rightChild.evaluate(X));
	}

	@Override
	public int treeDepth() {
		int leftDepth = leftChild.treeDepth();
        int rightDepth = rightChild.treeDepth();

        return Math.max(leftDepth, rightDepth) + 1;
	}

	@Override
	public int countNodes() {
		return 1 + leftChild.countNodes() + rightChild.countNodes();
	}

	@Override
	public String printExpression() {
		return "(" + leftChild.printExpression() + " " + Operators.getSymbol(binaryOperator) + " " + rightChild.printExpression() + ")";
	}

	@Override
	public Node copyTree(Node p) {
		BinaryNode n = new BinaryNode(binaryOperator, p);
		n.setLeftChild(leftChild.copyTree(n));
		n.setRightChild(rightChild.copyTree(n));
		
		return n;
	}

	@Override
	public Node getByIndex(int i) {
		int countLeft = this.leftChild.countNodes();
		
		if (i < countLeft) {
			return leftChild.getByIndex(i);
			
		} else if (countLeft == i) {
			return this;
			
		} else {
			return rightChild.getByIndex(i - countLeft - 1);
		}
	}

	public Node getParent() {
		return parent;
	}

	public void setParent(Node parent) {
		this.parent = parent;
	}

}
