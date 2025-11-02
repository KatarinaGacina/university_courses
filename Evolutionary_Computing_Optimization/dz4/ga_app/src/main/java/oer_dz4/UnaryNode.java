package oer_dz4;

import java.util.List;

public class UnaryNode extends Node {

	private IUnaryOperator unaryOperator;
	
	private Node parent;
	private Node child;
    
    public UnaryNode(IUnaryOperator o, Node parent) {
        this.unaryOperator = o;
        this.parent = parent;
        
        child = null;
    }

	IUnaryOperator getOperator() {
		return unaryOperator;
	}
	
	protected Node getChild() {
		return child;
	}
	
	protected void setChild(Node child) {
		this.child = child;
	}

	@Override
	public double evaluate(List<Double> X) {
		return unaryOperator.result(this.child.evaluate(X));
	}

	@Override
	public int treeDepth() {
		return child.treeDepth() + 1;
	}

	@Override
	public int countNodes() {
		return 1 + child.countNodes();
	}

	@Override
	public String printExpression() {
		return Operators.getSymbol(unaryOperator) + "(" + child.printExpression() + ")";
	}

	@Override
	public Node copyTree(Node p) {
		UnaryNode n = new UnaryNode(unaryOperator, p);
		n.setChild(child.copyTree(n));
		
		return n;
	}

	@Override
	public Node getByIndex(int i) {
		int maxIndex = child.countNodes();
		
        if (i < maxIndex) {
            return child.getByIndex(i);
            
        } else if (i == maxIndex) {
            return this;
            
        } else {
            return null;
        }
	}

	public Node getParent() {
		return parent;
	}

	public void setParent(Node parent) {
		this.parent = parent;
	}
}
