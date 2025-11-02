package oer.dz;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.linear.ArrayRealVector;

public class ClonAlg {
	
	private IFunction f;
	
	private Rjesenje[] populacija;
	private Rjesenje[] klonovi;
	
	private double beta = 80;
	//private double theta = 25;
	private int n = 20;
	private int N = 20;
	private int d = 2;
	private int max_iter = 1000;

	public ClonAlg(IFunction f) {
		this.f = f;
		this.populacija = new Rjesenje[N];
	}
	
	private void inicijaliziraj() {
		for (int i = 0; i < this.N; i++) {
			double[] rj = new double[this.f.getNumberOfVariables()];
			Random rand = new Random();
			for (int j = 0; j < this.f.getNumberOfVariables(); j++) {
	            rj[j] = -10 + 20 * rand.nextDouble(); //d_gran + (g_gran - d_gran) *
	        }
			this.populacija[i] = new Rjesenje(rj.clone(), this.f.getValue(new ArrayRealVector(rj)));
		}	
		
		int nc = 0;
		for (int i = 1; i <= n; i++) {
			nc += Math.floor(((this.beta * this.n) / i));
		}
		this.klonovi = new Rjesenje[nc];
	}
	
	private void kloniraj() {
		int k = 0;
 		for (int i = 1; i <= this.n; i++) { // odaberi n
			for (int j = 0; j < Math.floor(((this.beta * this.n) / i)); j++) {
				this.klonovi[k] = new Rjesenje(this.populacija[i - 1].getKoeficijenti().clone(), this.populacija[i - 1].getAfinitet());
				k++;
			}
		}
	}
	
	private void hipermutiraj(int zadrzi) {
		double p;
		//double fn;
		
		double[] rj = new double[this.f.getNumberOfVariables()];
		
		Random rand = new Random();
		
		for (int i = zadrzi; i < this.klonovi.length; i++) {
			rj = this.klonovi[i].getKoeficijenti();

			//fn = (pom.getAfinitet() - this.klonovi[0].getAfinitet()) / (this.klonovi[this.klonovi.length - 1].getAfinitet() - this.klonovi[0].getAfinitet());
			//p = Math.pow(Math.E, (- this.theta * fn));
			
			p = i / (double) this.klonovi.length;
			for (int j = 0; j < rj.length; j++) {
				if (rand.nextDouble() < p) {
					rj[j] += -1 + 2 * rand.nextDouble(); 
				}
			}
			
			this.klonovi[i].setKoeficijenti(rj.clone());
			this.klonovi[i].setAfinitet(this.f.getValue(new ArrayRealVector(rj)));
		}
	}
	
	public Rjesenje startAlgorithm() {
		inicijaliziraj();
		
		int t = 0;
		
		while (t < this.max_iter) {
			//afinitet rj. = vrednuj rješenja -> f.getValue
			Arrays.sort(this.populacija); //evaluacija rjesenja; sorted ascending
			
			kloniraj();
			hipermutiraj(1);
			
			Arrays.sort(this.klonovi);
		
			/*for (int z = 0; z < this.klonovi.length; z++) {
				System.out.println(Arrays.toString(this.klonovi[z].getKoeficijenti()));
			}
			System.out.println();*/
			
			for (int i = 0; i < this.N; i++) {
				this.populacija[i] = this.klonovi[i];
			}
			
			for (int i = (this.N - this.d - 1); i < this.N; i++) {
				double[] rj = new double[this.f.getNumberOfVariables()];
				
				Random rand = new Random();
				for (int j = 0; j < this.f.getNumberOfVariables(); j++) {
		            rj[j] = -10 + 20 * rand.nextDouble(); //d_gran + (g_gran - d_gran) *
		        }
				
				this.populacija[i] = new Rjesenje(rj, this.f.getValue(new ArrayRealVector(rj)));
			}
			
			t++;
		}
		
		Arrays.sort(this.populacija);
		
		/*for (int z = 0; z < this.populacija.length; z++) {
			System.out.println();
			System.out.println(Arrays.toString(this.populacija[z].getKoeficijenti()));
			System.out.println(this.populacija[z].getAfinitet());
		}*/
		
		return this.populacija[0];
	}

	public double getBeta() {
		return beta;
	}

	public void setBeta(double beta) {
		this.beta = beta;
	}

	public int getn() {
		return n;
	}

	public void setn(int n) {
		this.n = n;
	}

	public int getMax_iter() {
		return max_iter;
	}

	public void setMax_iter(int max_iter) {
		this.max_iter = max_iter;
	}
	
	int getD() {
		return d;
	}

	void setD(int d) {
		this.d = d;
	}
	
	int getN() {
		return N;
	}

	void setN(int N) {
		this.N = N;
	}

}
