package oer_dz4;

//import java.util.Stack;

public class Tree {
	 
	private Node root = null;
	
	public Tree() {	
	}
	
	public Tree(Node root) {	
		this.root = root;
	}

	
	public int getTreeDepth() {
		if (this.root == null) {
			return 0;
		}
		
        return this.root.treeDepth();
    }
	
	public int countInternalNodes(Node node) {
		if (this.root == null) {
			return 0;
		}
		
        return this.root.countNodes();
    }

}
