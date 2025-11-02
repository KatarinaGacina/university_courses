package oer.dz;

import java.io.Serializable;

public class Rjesenje implements Comparable<Rjesenje> {
	  
	  private double[] koeficijenti;
	  private double afinitet;

	  public Rjesenje(double[] koeficijenti, double afinitet) {
		  this.koeficijenti = koeficijenti;
		  this.afinitet = afinitet;
	  }
	
	  public int compareTo(Rjesenje other) {
		  if (this.afinitet < other.afinitet) {
			  return -1;
		  } else if (this.afinitet > other.afinitet) {
			  return +1;
		  }
	
		  return 0;
	  }
	
	  @Override
	  public boolean equals(Object obj) {
	       if (this == obj) {
	    	   return true;
	       }
	
	       if (obj == null || getClass() != obj.getClass()) {
	    	   return false;
	       }
	
	       Rjesenje other = (Rjesenje) obj;
	       
	       for (int i = 0; i < this.koeficijenti.length; i++) {
	    	   if (this.koeficijenti[i] != other.koeficijenti[i]) {
	    		   return false;
	    	   }
	       }
	       
	       return true;
	  }

	  public double[] getKoeficijenti() {
		  return this.koeficijenti;
	  }
	
	  public double getAfinitet() {
		  return this.afinitet;
	  }

	  void setKoeficijenti(double[] koeficijenti) {
		  this.koeficijenti = koeficijenti;
	  }

	  void setAfinitet(double afinitet) {
		  this.afinitet = afinitet;
	  }
	  
	  
}
