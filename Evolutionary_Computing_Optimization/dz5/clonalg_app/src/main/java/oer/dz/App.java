package oer.dz;

import java.util.Arrays;

public class App 
{
    public static void main( String[] args )
    {
    	String datoteka = args[0];
    	IFunction f4 = new Prijenosna(datoteka);
    	
    	ClonAlg ca = new ClonAlg(f4);
    	
    	Rjesenje rjesenje = ca.startAlgorithm();
    	System.out.println();
    	System.out.println("Rjesenje:");
    	System.out.println(Arrays.toString(rjesenje.getKoeficijenti()));
    	System.out.println("Pogreska:");
    	System.out.println(rjesenje.getAfinitet());
    }
}
