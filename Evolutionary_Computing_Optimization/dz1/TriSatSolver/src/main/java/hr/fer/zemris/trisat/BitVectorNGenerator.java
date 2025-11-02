package hr.fer.zemris.trisat;

import java.util.Iterator;

public class BitVectorNGenerator implements Iterable<MutableBitVector> {
	
	private BitVector vektor;
	
	public BitVectorNGenerator(BitVector rjesenje) {
		this.vektor = rjesenje;
	}
	
	// Vraća lijeni iterator koji na svaki next() računa sljedećeg susjeda
	@Override
	public Iterator<MutableBitVector> iterator() {
		return null;
	}
	
	// Vraća kompletno susjedstvo kao jedno polje
	public MutableBitVector[] createNeighborhood() {
		MutableBitVector[] susjedstvo = new MutableBitVector[this.vektor.getSize()];
		
		for (int i = 0; i < this.vektor.getSize(); i++) {
			boolean old_value = this.vektor.get(i);
			MutableBitVector permutation = this.vektor.copy();
            
            permutation.set(i, !old_value);
            susjedstvo[i] = permutation;
        }
		
		return susjedstvo;
	}

}
