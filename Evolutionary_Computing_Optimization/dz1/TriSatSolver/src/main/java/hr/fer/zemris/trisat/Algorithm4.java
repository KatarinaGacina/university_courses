package hr.fer.zemris.trisat;

//import java.util.ArrayList;
import java.util.Optional;
import java.util.Random;

//GSAT
public class Algorithm4 extends Algorithm implements IOptAlgoritam {
	
	private SATFormula formula;
	
	private int MAX_TRIES = 2;
	//private int MAX_FLIPS = 100000;
	
	public Algorithm4(SATFormula formula) {
		this.formula = formula;
	}

	/**
	public Optional<BitVector> solve2(Optional<BitVector> initial) {
		Optional<BitVector> return_result = null;
		
		int t = 0;
		while (t < MAX_TRIES) {
			BitVector rjesenje = new BitVector(new Random(), this.formula.getNumberOfVariables());
			
			if (fitness_original(rjesenje, this.formula) == this.formula.getNumberOfClauses()) {
				return Optional.of(rjesenje);
			}
			
			MutableBitVector rjesenje_m = rjesenje.copy();
			
			int flip = 0;
			while (flip < MAX_FLIPS) {
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
					return return_result = Optional.of(rez);
				}
				
				flip = flip + 1;			
			}
			
			t = t + 1;
		}
		
		return return_result;
	}*/
	
	
	@Override
	public Optional<BitVector> solve(Optional<BitVector> initial) {
		Optional<BitVector> return_result = null;
		
		int t = 0;
		while (t < MAX_TRIES) {
			Algorithm3 a3 = new Algorithm3(this.formula);
			a3.setMaxFlips(1000);
			BitVector pocetno = new BitVector(new Random(), this.formula.getNumberOfVariables());

			return_result = a3.solve(Optional.of(pocetno));
			
			if (return_result != null) {
				return return_result;
			}
			
			t = t + 1;
		}
		
		return return_result;
		
	}

}
