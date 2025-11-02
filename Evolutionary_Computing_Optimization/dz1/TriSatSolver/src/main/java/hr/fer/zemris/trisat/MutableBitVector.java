package hr.fer.zemris.trisat;

public class MutableBitVector extends BitVector {
	
	public MutableBitVector(boolean... bits) {
		super(bits);
	}
	
	public MutableBitVector(int n) {
		super(n);
	}
	
	// zapisuje predanu vrijednost u zadanu varijablu
	public void set(int index, boolean value) {
		this.getBits()[index] = value;
	}
}
