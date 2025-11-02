package hr.fer.zemris.trisat;

import java.util.Optional;

public class Algorithm1 extends Algorithm implements IOptAlgoritam {
	
	private SATFormula formula;
	
	public Algorithm1(SATFormula formula) {
		this.formula = formula;
	}
	
	public boolean checkSolution(MutableBitVector vektor, SATFormula formula) {
		boolean ukupna_vrijednost_klauzula = true;
		
		for (int c = 0; c < formula.getNumberOfClauses(); c++) {
			Clause clause = formula.getClause(c);
			
			boolean vrijednost_klauzule = checkClause(vektor, clause);
			
			if (!vrijednost_klauzule) {
				ukupna_vrijednost_klauzula = false;
				break;
			}
		}
			
		return ukupna_vrijednost_klauzula;
	}

	@Override
	public Optional<BitVector> solve(Optional<BitVector> initial) {
		Optional<BitVector> return_result = null;
		
		//iscrpno pretraživanje
		
		BitVector vektor = new BitVector(this.formula.getNumberOfVariables());
		MutableBitVector m_vektor = vektor.copy();
		
		int numCombinations = (int) Math.pow(2, this.formula.getNumberOfVariables());
		
		for (int i = 0; i < numCombinations; i++) {
			
            for (int j = 0; j < this.formula.getNumberOfVariables(); j++) {
                m_vektor.set(j, ((i >> j) & 1) == 1);
            }
            
            if (checkSolution(m_vektor, this.formula)) {
            	BitVector rez = m_vektor;
				return_result = Optional.of(rez);
            	System.out.println(m_vektor.toString());
            }
        }
		
		return return_result;
	}

}
