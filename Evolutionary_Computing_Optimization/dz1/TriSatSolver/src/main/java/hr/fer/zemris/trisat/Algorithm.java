package hr.fer.zemris.trisat;

public class Algorithm {
	
	public static boolean checkClause(BitVector vektor, Clause clause) {
		boolean ukupna_vrijednost_klauzule = false;

		for (int i = 0; i < clause.getSize(); i++) {
			int index = clause.getLiteral(i);
			
			boolean vrijednost;

			if (index < 0) {
				vrijednost = !vektor.get((-index) - 1);
			} else {
				vrijednost = vektor.get(index - 1);
			}
			
			if (vrijednost) {
				ukupna_vrijednost_klauzule = true;
				break;
			}
		}
			
		return ukupna_vrijednost_klauzule;
	}
	
	public static int fitness_original(BitVector v, SATFormula formula) {
		int n = 0;
		
		for (int c = 0; c < formula.getNumberOfClauses(); c++) {
			Clause clause = formula.getClause(c);
			
			boolean vrijednost_klauzule = checkClause(v, clause);
			
			if (vrijednost_klauzule) {
				n++;
			}
		}
			
		return n;
	}

}
