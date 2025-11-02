package hr.fer.zemris.trisat;

import java.util.ArrayList;
import java.util.Optional;
import java.util.Random;

public class Algorithm2 extends Algorithm implements IOptAlgoritam {
	
	private SATFormula formula;
	private int max_br_iter = 100000;
	
	public Algorithm2(SATFormula formula) {
		this.formula = formula;
	}

	@Override
	public Optional<BitVector> solve(Optional<BitVector> initial) {
		Optional<BitVector> return_result = null;
		
		//Iterativni algoritam pretraživanja
		
		//funkcija dobrote -> broj klauzula koje rjesenje zadovoljava
		
		BitVector rjesenje;
		if (initial == null) {
			rjesenje = new BitVector(new Random(), this.formula.getNumberOfVariables());
		} else {
			rjesenje = initial.get();
		}
		
		if (fitness_original(rjesenje, this.formula) == this.formula.getNumberOfClauses()) {
			return Optional.of(rjesenje);
		}
		
		MutableBitVector rjesenje_m = rjesenje.copy();
		
		int t = 0;
		while (t < max_br_iter) {
			BitVectorNGenerator generator = new BitVectorNGenerator(rjesenje_m);
			MutableBitVector[] susjedstvo = generator.createNeighborhood();
			
			int trenutni_fit = fitness_original(rjesenje_m, this.formula);
			
			ArrayList<MutableBitVector> fit_rjesenja = new ArrayList<>();
			for (MutableBitVector rj : susjedstvo) {
				int fit = fitness_original(rj, this.formula);
				
				if (trenutni_fit < fit) {
					fit_rjesenja.add(rj);
				}
			}
			
			if (fit_rjesenja.size() < 1) {
				break;
				
			} else {
				Random choose_next = new Random();
				rjesenje_m = fit_rjesenja.get(choose_next.nextInt(fit_rjesenja.size()));
			}
			
			if (fitness_original(rjesenje_m, this.formula) == this.formula.getNumberOfClauses()) {
				BitVector rez = rjesenje_m;
				return_result = Optional.of(rez);
				break;
			}
			
			t = t + 1;
			
		}
		
		return return_result;
	}

}
