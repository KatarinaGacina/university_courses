package hr.fer.zemris.trisat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

public class Algorithm5 extends Algorithm implements IOptAlgoritam {
	
	private SATFormula formula;
	
	private int MAX_FLIPS = 1000;
	private int MAX_TRIES = 2;
	
	private double p = 0.3;
	
	public Algorithm5(SATFormula formula) {
		this.formula = formula;
	}

	@Override
	public Optional<BitVector> solve(Optional<BitVector> initial) {
		Optional<BitVector> return_result = null;
		
		int tries = 0;
		while (tries < MAX_TRIES) {
			BitVector rjesenje = new BitVector(new Random(), this.formula.getNumberOfVariables());
			
			if (fitness_original(rjesenje, this.formula) == this.formula.getNumberOfClauses()) {
				return Optional.of(rjesenje);
			}
			
			MutableBitVector rjesenje_m = rjesenje.copy();
			
			int flip = 0;
			
			ArrayList<Clause> nezadovoljene_klauzule = new ArrayList<>();
			Random rand = new Random();
			
			while (flip < MAX_FLIPS) {
				for (int i = 0; i < this.formula.getNumberOfClauses(); i++) {
					Clause clause = this.formula.getClause(i);
					
					if (!clause.isSatisfied(rjesenje_m)) {
						nezadovoljene_klauzule.add(this.formula.getClause(i));
					}
				}
				
				Clause klauzula = nezadovoljene_klauzule.get(rand.nextInt(nezadovoljene_klauzule.size()));
				
				if (Math.random() < p) {
					int indexFlip = rand.nextInt(klauzula.getSize());
					int index = klauzula.getLiteral(indexFlip);
					
					if (index < 0) {
						index = -index;
					}
					index -= 1;

					rjesenje_m.set(index, !rjesenje.get(index));
				
				} else {
					Map<MutableBitVector, Double> mapa = new HashMap<>();
					
					BitVectorNGenerator generator = new BitVectorNGenerator(rjesenje_m);
					MutableBitVector[] susjedstvo = generator.createNeighborhood();
					
					for (MutableBitVector rj : susjedstvo) {
						double fit = fitness_original(rj, this.formula); 
						mapa.put(rj, fit);
					}
					
					rjesenje_m = Collections.max(mapa.entrySet(), Map.Entry.comparingByValue()).getKey();
					
					mapa.clear();
				}
				
				if (fitness_original(rjesenje_m, this.formula) == this.formula.getNumberOfClauses()) {
					return Optional.of(rjesenje_m);
				}
				
				flip = flip + 1;		
				nezadovoljene_klauzule.clear();
			}
			
			tries += 1;
		}
		
		return return_result;
	}

}
