package hr.fer.zemris.trisat;

import java.util.Arrays;

public class Clause {
	
	private int[] indexes;
	
	public Clause(int[] indexes) {
		this.indexes = indexes;
	}
	
	// vraća broj literala koji čine klauzulu
	public int getSize() {
		return this.indexes.length;
	}
	
	// vraća indeks varijable koja je index-ti član ove klauzule
	public int getLiteral(int index) {
		return indexes[index];
	}
	
	// vraća true ako predana dodjela zadovoljava ovu klauzulu
	public boolean isSatisfied(BitVector assignment) {
		boolean ukupna_vrijednost_klauzule = false;

		for (int i = 0; i < this.getSize(); i++) {
			int index = this.getLiteral(i);
			
			boolean vrijednost;

			if (index < 0) {
				vrijednost = !assignment.get((-index) - 1);
			} else {
				vrijednost = assignment.get(index - 1);
			}
			
			if (vrijednost) {
				ukupna_vrijednost_klauzule = true;
				break;
			}
		}
			
		return ukupna_vrijednost_klauzule;
	}
	
	@Override
	public String toString() {
		return Arrays.toString(indexes);
	}
}
