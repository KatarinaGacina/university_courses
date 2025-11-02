package hr.fer.zemris.optjava.dz2;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Sustav implements IFunction {
	private String datoteka;
	
	private RealMatrix X;
	private RealVector Y;

	public Sustav(String datoteka) {
		this.datoteka = datoteka;
		
        this.X = null;
        this.Y = null;
		
		try (BufferedReader reader = Files.newBufferedReader(Paths.get(datoteka))) {
            String line;
            
            int rowNum = 0;
            while ((line = reader.readLine()) != null) {
                if (!line.startsWith("#")) {
                	String[] n_n = line.trim().substring(1, line.length() - 1).split(", ");

                	if (this.X == null)  {
                		this.X = new Array2DRowRealMatrix(n_n.length - 1, n_n.length - 1);
                		this.Y = new ArrayRealVector(n_n.length - 1);
                	}
                	
                    for (int i = 0; i < n_n.length - 1; i++) {
                        X.addToEntry(rowNum, i, Double.parseDouble(n_n[i]));
                    }
                    Y.addToEntry(rowNum, Double.parseDouble(n_n[n_n.length - 1]));
                    
                    rowNum++;
                }
            }
            
        } catch (IOException e) {
            e.printStackTrace();
        }
	}

	public String getDatoteka() {
		return datoteka;
	}

	@Override
	public int getNumberOfVariables() {
		return this.Y.getDimension();
	}
	
	
	@Override
	public double getValue(RealVector x) {
		return ((this.X.operate(x)).subtract(this.Y)).getNorm();
	}

	@Override
	public RealVector getGradient(RealVector x, boolean neg) {
		RealVector gradient = this.X.transpose().operate(this.X.operate(x).subtract(this.Y)).mapMultiply(2);
        
        if (neg) {
        	return gradient.mapMultiplyToSelf(-1);
        }
        
        return gradient;
	}
	
	public double error(RealVector x) {
		RealVector r = (this.X.operate(x)).subtract(this.Y);
			
		return r.dotProduct(r);
	}
	
	public static void main(String[] args) {
		if (args.length < 2) {
			throw new IllegalArgumentException("Insufficient number of arguments provided.");
		}
		
		int maxIter = Integer.parseInt(args[0]);
		String datoteka = args[1];
		
		NumOptAlgorithms opt = new NumOptAlgorithms();
		
		Sustav s = new Sustav(datoteka);

		RealVector rj = opt.gradijentniSpust(s, maxIter, null);
		System.out.println(s.error(rj));
	}
	
}
