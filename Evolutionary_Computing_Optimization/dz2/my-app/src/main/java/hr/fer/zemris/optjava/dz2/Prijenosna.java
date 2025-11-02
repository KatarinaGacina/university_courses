package hr.fer.zemris.optjava.dz2;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Prijenosna implements IFunction {
	
	private String datoteka;
	
	private RealMatrix X;
	private RealVector Y;
	
	double scale = 1e-8;

	public Prijenosna(String datoteka) {
		this.datoteka = datoteka;
		
        this.X = null;
        this.Y = null;
		
		try (BufferedReader reader = Files.newBufferedReader(Paths.get(datoteka))) {
            String line;
            
            List<double[]> X_p = new ArrayList<>();
            List<Double> Y_p = new ArrayList<>();
            
            while ((line = reader.readLine()) != null) {
                if (!line.startsWith("#")) {
                	String[] n_n = line.trim().substring(1, line.length() - 1).split(", ");
                	
                    double[] row = new double[n_n.length - 1];
                    for (int i = 0; i < n_n.length - 1; i++) {
                        row[i] = Double.parseDouble(n_n[i]);
                    }
                    X_p.add(row);

                    Y_p.add(Double.parseDouble(n_n[n_n.length - 1]));
                }
            }
            
            this.X = new Array2DRowRealMatrix(X_p.toArray(new double[X_p.size()][]));
            this.Y = new ArrayRealVector(Y_p.toArray(new Double[Y_p.size()]));
            
        } catch (IOException e) {
            e.printStackTrace();
        }
	}

	@Override
	public int getNumberOfVariables() {
		return 6;
	}
	
	private double getNumber(RealVector a, int k) {
		int N = this.X.getRowDimension();
		
		double suma = 0;
		for (int i = 0; i < N; i++) {
			double[] row = this.X.getRow(i);
			
			//y( x1, x2 , x3 , x4 , x5)=a⋅x1+b⋅x1 3 x2+ced⋅x3(1+c o s( e⋅x4))+f⋅x4 x5 2 
			double y_rj = a.getEntry(0) * row[0] + a.getEntry(1) * Math.pow(row[0], 3) * row[1] + a.getEntry(2) * Math.pow(Math.E, (a.getEntry(3) * row[2]))*(1 + Math.cos(a.getEntry(4) * row[3])) + a.getEntry(5) * row[3] * Math.pow(row[4], 2);
			
			if (k == 0) {
				suma += Math.pow((y_rj - this.Y.getEntry(i)), 2);
			} else if (k == 1) {
				suma += scale * (y_rj - this.Y.getEntry(i)) * row[0];
			} else if (k == 2) {
				suma += scale * (y_rj - this.Y.getEntry(i)) * Math.pow(row[0], 3) * row[1];
			} else if (k == 3) {
				suma += scale * (y_rj - this.Y.getEntry(i)) * Math.pow(Math.E, (a.getEntry(3) * row[2]))*(1 + Math.cos(a.getEntry(4)) * row[3]);
			} else if (k == 4) {
				suma += scale * (y_rj - this.Y.getEntry(i)) * a.getEntry(2) * Math.pow(Math.E, a.getEntry(3) * row[2]) * (1 + Math.cos(a.getEntry(4) * row[3])) * row[2];
			} else if (k == 5) {
				suma += scale * (y_rj - this.Y.getEntry(i)) * a.getEntry(2) * Math.pow(Math.E, a.getEntry(3) * row[2]) * (- Math.sin(a.getEntry(4) * row[3])) * row[3];
			} else if (k == 6) {
				suma += scale * (y_rj - this.Y.getEntry(i)) * row[3] * Math.pow(row[4], 2);
			}
		}
		
		return suma;
	}

	@Override
	public double getValue(RealVector a) {
		return this.getNumber(a, 0) * 1/2;
	}

	@Override
	public RealVector getGradient(RealVector a, boolean neg) {		
		RealVector gradient = new ArrayRealVector(this.getNumberOfVariables());
		
		gradient.setEntry(0, this.getNumber(a, 1));
		gradient.setEntry(1, this.getNumber(a, 2));
		gradient.setEntry(2, this.getNumber(a, 3));
		gradient.setEntry(3, this.getNumber(a, 4));
		gradient.setEntry(4, this.getNumber(a, 5));
		gradient.setEntry(5, this.getNumber(a, 6));

		if (neg) {
			return gradient.mapMultiply(-1);
		} else {
			return gradient;
		}
	}
	
	public static void main(String[] args) {
		if (args.length < 2) {
			throw new IllegalArgumentException("Insufficient number of arguments provided.");
		}
		
		int maxIter = Integer.parseInt(args[0]);
		String datoteka = args[1];
		
		NumOptAlgorithms opt = new NumOptAlgorithms();
		
		Prijenosna p = new Prijenosna(datoteka);

		opt.gradijentniSpust(p, maxIter, null);
		
	}

	public String getDatoteka() {
		return datoteka;
	}

}
