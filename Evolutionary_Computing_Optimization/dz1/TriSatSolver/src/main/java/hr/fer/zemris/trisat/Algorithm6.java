package hr.fer.zemris.trisat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Random;

public class Algorithm6 extends Algorithm implements IOptAlgoritam {
	
	private SATFormula formula;
	
	private int MAX_TRIES = 2;
	private int MAX_FLIPS = 1000;
	
	private double p = 0.5;
	
	public Algorithm6(SATFormula formula) {
		this.formula = formula;
	}
	
	public Optional<BitVector> solve(Optional<BitVector> initial) {
		Optional<BitVector> return_result = null;
		
		BitVector vektor = new BitVector(new Random(), this.formula.getNumberOfVariables());
		if (fitness_original(vektor, this.formula) == this.formula.getNumberOfClauses()) {
			BitVector rez = vektor;
			return return_result = Optional.of(rez);
		}
		
		MutableBitVector rjesenje = vektor.copy();
		
		int t = 0;
		while (t < MAX_TRIES) {
			
			int flip = 0;
			while (flip < MAX_FLIPS) {
				BitVectorNGenerator generator = new BitVectorNGenerator(rjesenje);
				MutableBitVector[] susjedstvo = generator.createNeighborhood();
				
				int trenutni_fit = fitness_original(rjesenje, this.formula);
				
				ArrayList<MutableBitVector> fit_rjesenja = new ArrayList<>();
				for (MutableBitVector rj : susjedstvo) {
					int fit = fitness_original(rj, this.formula);
					
					if (trenutni_fit < fit) {
						fit_rjesenja.add(rj);
					}
				}
				
				if (fit_rjesenja.size() < 1) {
					int changeNum = Math.max(1, (int) (rjesenje.getSize() * p));
					
			        List<Integer> indices = new ArrayList<>();
			        for (int i = 0; i < rjesenje.getSize(); i++) {
			            indices.add(i);
			        }
			        Collections.shuffle(indices);

			        for (int i = 0; i < changeNum; i++) {
			            int index = indices.get(i);
			            rjesenje.set(index, !rjesenje.get(i));
			        }
			        
					break;
					
				} else {
					Random choose_next = new Random();
					rjesenje = fit_rjesenja.get(choose_next.nextInt(fit_rjesenja.size()));
				}
				
				if (fitness_original(rjesenje, this.formula) == this.formula.getNumberOfClauses()) {
					BitVector rez = rjesenje;
					return return_result = Optional.of(rez);
				}
				
				flip = flip + 1;
				
			}
			
			t = t + 1;
		}
		
		return return_result;
		
	}

}
