package hr.fer.zemris.trisat;

import java.util.Arrays;
import java.util.Random;

public class BitVector {
	
	private boolean[] bits;
	
	boolean[] getBits() {
		return bits;
	}

	public BitVector(Random rand, int numberOfBits) {
		bits = new boolean[numberOfBits];
		
		for (int i = 0; i < numberOfBits; i++) {
            bits[i] = rand.nextBoolean();
        }
	}
	
	public BitVector(boolean ... bits) {
		this.bits = bits;
	}
	
	public BitVector(int n) {
		bits = new boolean[n];
        Arrays.fill(bits, false);
	}
	
	// vraća vrijednost index-te varijable
	public boolean get(int index) {
		return bits[index];
	}
	
	// vraća broj varijabli koje predstavlja
	public int getSize() {
		return bits.length;
	}
	
	@Override
	public String toString() {
		StringBuilder prikaz = new StringBuilder();
		
		for (boolean b : bits) {
			if (b == true) {
				prikaz.append("1");
			} else {
				prikaz.append("0");
			}
		}
		
		return prikaz.toString();
	}
	
	// vraća promjenjivu kopiju trenutnog rješenja
	public MutableBitVector copy() {
		return new MutableBitVector(Arrays.copyOf(this.bits, this.bits.length));
	}
}
