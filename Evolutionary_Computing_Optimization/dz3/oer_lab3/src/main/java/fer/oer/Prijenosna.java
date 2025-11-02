package fer.oer;

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
	
	@Override
	public double getValue(RealVector a) {
		int N = this.X.getRowDimension();
		
		double suma = 0;
		for (int i = 0; i < N; i++) {
			double[] row = this.X.getRow(i);
			
			//y( x1, x2 , x3 , x4 , x5)=a⋅x1+b⋅x1 3 x2+ced⋅x3(1+c o s( e⋅x4))+f⋅x4 x5 2 
			double y_rj = a.getEntry(0) * row[0] + a.getEntry(1) * Math.pow(row[0], 3) * row[1] + a.getEntry(2) * Math.pow(Math.E, (a.getEntry(3) * row[2]))*(1 + Math.cos(a.getEntry(4) * row[3])) + a.getEntry(5) * row[3] * Math.pow(row[4], 2);
			suma += Math.pow((y_rj - this.Y.getEntry(i)), 2);
		}

		return suma * 1/2;
	}

	public String getDatoteka() {
		return datoteka;
	}

}
