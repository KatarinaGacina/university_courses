package hr.fer.zemris.trisat;

import java.util.Optional;

public interface IOptAlgoritam {
	Optional<BitVector> solve(Optional<BitVector> initial);
}
