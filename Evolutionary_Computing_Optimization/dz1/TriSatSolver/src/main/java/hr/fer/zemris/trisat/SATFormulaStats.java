package hr.fer.zemris.trisat;

/**
public class SATFormulaStats {
	
	private SATFormula formula;
	
	private BitVector rjesenje = null;
	
	private double[] post;
	
	private double percentageConstantUp = 0.01;
	private double percentageConstantDown = 0.1;
	private int numberOfBest = 2;
	private double percentageUnitAmount = 50;
	
	public SATFormulaStats(SATFormula formula) {
		this.formula = formula;
	}
	
	// analizira se predano rješenje i pamte svi relevantni pokazatelji
	// primjerice, ažurira elemente polja post[...] ako drugi argument to dozvoli; računa Z; ...
	public void setAssignment(BitVector assignment, boolean updatePercentages) {
		this.rjesenje = assignment;
		
		if (updatePercentages) {
			
		}
	}
	
	// vraća temeljem onoga što je setAssignment zapamtio: broj klauzula koje su zadovoljene
	public int getNumberOfSatisfied() {
		if (this.rjesenje == null) {
			throw new IllegalArgumentException();
		}
			
		int n = 0;
		
		for (int c = 0; c < formula.getNumberOfClauses(); c++) {
			Clause clause = formula.getClause(c);
			
			boolean vrijednost_klauzule = Algorithm.checkClause(this.rjesenje.copy(), clause);
			
			if (vrijednost_klauzule) {
				n++;
			}
		}
			
		return n;
	}
	
	// vraća temeljem onoga što je setAssignment zapamtio
	public boolean isSatisfied() {
		return (this.getNumberOfSatisfied() == this.formula.getNumberOfClauses());
	}
	
	// vraća temeljem onoga što je setAssignment zapamtio: suma korekcija klauzula
	// to je korigirani Z iz algoritma 3
	public double getPercentageBonus() {
		return 0;
	}
	
	// vraća temeljem onoga što je setAssignment zapamtio: procjena postotka za klauzulu
	// to su elementi polja post[...]
	public double getPercentage(int index) {
		return this.post[index];
	}
	
	// resetira sve zapamćene vrijednosti na početne (tipa: zapamćene statistike)
	public void reset() {
		Arrays.fill(this.post, 0);
	}
}*/
