package hr.fer.zemris.trisat;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.stream.Collectors;

public class Algorithm3 extends Algorithm implements IOptAlgoritam {
	
	private double percentageConstantUp = 0.01;
	private double percentageConstantDown = 0.1;
	private int numberOfBest = 2;
	private double percentageUnitAmount = 50;
	
	private SATFormula formula;
	private int max_br_iter = 100000;
	
	private double[] post;
	
	public Algorithm3(SATFormula formula) {
		this.formula = formula;
		
		this.post = new double[this.formula.getNumberOfClauses()];
		Arrays.fill(this.post, 0);
	}
	
	public void setMaxFlips(int num) {
		this.max_br_iter = num;
	}
	
	private void post_initial(BitVector rjesenje) {
		for (int c = 0; c < formula.getNumberOfClauses(); c++) {
			Clause clause = formula.getClause(c);
			
			boolean vrijednost_klauzule = checkClause(rjesenje, clause);
			
			if (vrijednost_klauzule) {
				this.post[c] = (1 - post[c]) * this.percentageConstantUp;
			} else {
				this.post[c] = (0 - post[c]) * this.percentageConstantDown;
			}
		}
	}
	
	private double fitnessZ(BitVector susjed, double Z) {
		for (int c = 0; c < formula.getNumberOfClauses(); c++) {
			Clause clause = formula.getClause(c);
			
			boolean vrijednost_klauzule = checkClause(susjed, clause);
			
			if (vrijednost_klauzule) {
				Z =  Z + ((1 - post[c]) * this.percentageUnitAmount);
			} else {
				Z =  Z + ((1 - post[c]) * -this.percentageUnitAmount);
			}
		}
		
		return Z;
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
		
		Map<MutableBitVector, Double> mapa = new HashMap<>();
		
		int t = 0;
		while (t < max_br_iter) {
			post_initial(rjesenje_m);
			
			BitVectorNGenerator generator = new BitVectorNGenerator(rjesenje_m);
			MutableBitVector[] susjedstvo = generator.createNeighborhood();
			
			for (MutableBitVector rj : susjedstvo) {
				double fit = fitness_original(rj, this.formula);
				double Z = fitnessZ(rj, fit); 
				
				mapa.put(rj, Z);
			}
			
			Random choose_next = new Random();
			List<MutableBitVector> topSolutions = mapa.entrySet().stream()
	                .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
	                .limit(this.numberOfBest)
	                .map(Map.Entry::getKey)
	                .collect(Collectors.toList());
			
			rjesenje_m = topSolutions.get(choose_next.nextInt(this.numberOfBest));
			
			if (fitness_original(rjesenje_m, this.formula) == this.formula.getNumberOfClauses()) {
				BitVector rez = rjesenje_m;
				return_result = Optional.of(rez);
				break;
			}
			
			t = t + 1;
			mapa.clear();
		}
		
		return return_result;
	}

}
