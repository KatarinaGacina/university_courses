package fer.oer; 

import java.util.Arrays;
//import java.util.HashSet;
import java.util.Random;
//import java.util.Set;

import org.apache.commons.math3.linear.ArrayRealVector;

public class SimulatedAnnealing {
	private double g_gran = 0;
	private double d_gran = 5;
	//private double n_interval = 0.1;
	
	private double rjesenje[];
	private double best_solution[];
	
	private boolean maximize;
	private IFunction function;
	
	private double alpha = 0.99;
	private double T = 100;
	private int max_iter = 50;
	private int ekv = 50000;
	
	private double min_T = 0.00001;
	
	public double geometricCooling(double t0, double alpha, int i) {
		return t0 * Math.pow(alpha, i);
	}
	
	private double[] generirajSusjeda(double s) {
		double[] neighbour = Arrays.copyOf(this.rjesenje, this.function.getNumberOfVariables());
		
		Random rand = new Random();
	    int index = rand.nextInt(neighbour.length);
	    neighbour[index] += rand.nextGaussian() * s;
		
		return neighbour;
	}
	
	/**private double[] generirajSusjeda(int n) {
		double[] neighbour = Arrays.copyOf(this.rjesenje, this.function.getNumberOfVariables());
		
		Set<Integer> indexes = new HashSet<Integer>();
		Random rand = new Random();
		
		while (indexes.size() < n) {
			indexes.add(rand.nextInt(neighbour.length));
		}
		
		for (Integer i : indexes) {
			neighbour[i] += -n_interval + (n_interval - (-n_interval)) * rand.nextDouble();
        }

		return neighbour;
	}*/

	public SimulatedAnnealing(boolean maximize, IFunction f) {
		this.maximize = maximize;
		this.function = f;
		
		int n = f.getNumberOfVariables();
		this.rjesenje = new double[n];
	}
	
	public double[] startAlgorithm() {
		Random rand = new Random();
		for (int i = 0; i < this.function.getNumberOfVariables(); i++) {
            rjesenje[i] = d_gran + (g_gran - d_gran) * rand.nextDouble();
        }
		this.best_solution = Arrays.copyOf(this.rjesenje, this.function.getNumberOfVariables());
		 
		int i = 0;
		while (i < this.max_iter && this.T > this.min_T) {
			int k = 0;
			while (k < this.ekv) {
				double[] neighbour = this.generirajSusjeda(1);
				
				double dE = this.function.getValue(new ArrayRealVector(neighbour)) - this.function.getValue(new ArrayRealVector(this.rjesenje));
				
				if (this.maximize) {
					dE *= (double)(-1);
				}
				
				if (dE < 0) {
					this.rjesenje = this.best_solution = neighbour;
					
				} else {
					double p = Math.exp(-dE / this.T);
					
					if (Math.random() <= p) {
						this.rjesenje = neighbour;
					}
				}
				
				k++;
			}
			
			System.out.println(Arrays.toString(best_solution) + " " + Arrays.toString(this.rjesenje) + " " + String.valueOf(this.function.getValue(new ArrayRealVector(this.rjesenje))));
			
			this.T = this.geometricCooling(this.T, this.alpha, i);
			
			i++;
		}
		
		
		return this.best_solution;
		
	}

	double getG_gran() {
		return g_gran;
	}

	void setG_gran(double g_gran) {
		this.g_gran = g_gran;
	}

	double getD_gran() {
		return d_gran;
	}

	void setD_gran(double d_gran) {
		this.d_gran = d_gran;
	}

	double getAlpha() {
		return alpha;
	}

	void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	double getT() {
		return T;
	}

	void setT(double t) {
		T = t;
	}

	int getMax_iter() {
		return max_iter;
	}

	void setMax_iter(int max_iter) {
		this.max_iter = max_iter;
	}

	int getEkv() {
		return ekv;
	}

	void setEkv(int ekv) {
		this.ekv = ekv;
	}

	double getMin_T() {
		return min_T;
	}

	void setMin_T(double min_T) {
		this.min_T = min_T;
	}
	
}
