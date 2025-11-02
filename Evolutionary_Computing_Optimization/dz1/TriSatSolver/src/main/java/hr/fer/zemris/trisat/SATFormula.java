package hr.fer.zemris.trisat;

public class SATFormula {
	
	private int numberOfVariables;
	private Clause[] clauses;
	
	public SATFormula(int numberOfVariables, Clause[] clauses) {
		this.numberOfVariables = numberOfVariables;
		this.clauses = clauses;
	}
	
	public int getNumberOfVariables() {
		return this.numberOfVariables;
	}
	
	public int getNumberOfClauses() {
		return this.clauses.length;
	}
	
	public Clause getClause(int index) {
		return clauses[index];
	}
	
	public boolean isSatisfied(BitVector assignment) {
		return false;
	}
	
	@Override
	public String toString() {
		return null;
	}
}
