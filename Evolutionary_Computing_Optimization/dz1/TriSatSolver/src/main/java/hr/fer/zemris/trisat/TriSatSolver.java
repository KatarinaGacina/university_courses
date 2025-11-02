package hr.fer.zemris.trisat;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;

public class TriSatSolver {
	
	private static void rezultat(Optional<BitVector> rez, boolean print) {
		if (rez == null) {
			System.out.println("Neuspjeh!");
		} else {
			if (print) {
				System.out.println(rez.get());
			}
			System.out.println("Uspjeh!");
		}
	}
	
	public static void main(String[] args) {
		if (args.length < 2) {
			throw new IllegalArgumentException("Insufficient number of arguments provided.");
		}
		
		String algorithm_index = args[0];
		String datoteka = args[1];
		
		Path sourcePath = Paths.get(datoteka);
		
		int numberOfVariables = 0;
		List<Clause> clauseList = new ArrayList<>();
		 
		try (BufferedReader reader = Files.newBufferedReader(sourcePath)) {
			 String line;
			 String[] components;
			 
	         while ((line = reader.readLine()) != null) {
	        	 
	        	 if (line.startsWith("%")) {
	        		 break;
	        		 
	        	 } else if (line.startsWith("p")) {
	        		 components = line.trim().split("\\s+");
	        		 numberOfVariables = Integer.parseInt(components[2]);
	        		 // int numberOfClauses = Integer.parseInt(components[3]);
	        		 
	        	 } else if (!line.startsWith("c")) {
	        		 //System.out.println(line);
	        		 
	        		 components = line.trim().split("\\s+");
	        		 int[] indexes = new int[components.length - 1];
	        		 for (int i = 0; i < indexes.length; i++) {	        		
	        			 indexes[i] = Integer.parseInt(components[i]);
	        		 }
	        		 //System.out.println(Arrays.toString(indexes));
	        		 
	                 clauseList.add(new Clause(indexes));

	        	 } 
	         }
	    } catch (IOException e) {
	    	 e.printStackTrace();
	    }
		
		//System.out.println(clauseList.toArray(new Clause[0]).length);
		SATFormula sat = new SATFormula(numberOfVariables, clauseList.toArray(new Clause[0]));
		
		if (algorithm_index.equals("1")) {
			Algorithm1 a1 = new Algorithm1(sat);
			
			Optional<BitVector> rez = a1.solve(null);
			rezultat(rez, false);
			
			
		} else if (algorithm_index.equals("2")) {
			Algorithm2 a2 = new Algorithm2(sat);
			BitVector pocetno = new BitVector(new Random(), numberOfVariables);

			Optional<BitVector> rez = a2.solve(Optional.of(pocetno));
			rezultat(rez, true);
			
		} else if (algorithm_index.equals("3")) {
			Algorithm3 a3 = new Algorithm3(sat);
			BitVector pocetno = new BitVector(new Random(), numberOfVariables);

			Optional<BitVector> rez = a3.solve(Optional.of(pocetno));
			rezultat(rez, true);
			
		} else if (algorithm_index.equals("4")) {
			Algorithm4 a4 = new Algorithm4(sat);

			Optional<BitVector> rez = a4.solve(null);
			rezultat(rez, true);
			
		} else if (algorithm_index.equals("5")) {
			Algorithm5 a5 = new Algorithm5(sat);
			
			Optional<BitVector> rez = a5.solve(null);
			rezultat(rez, true);
			
		} else if (algorithm_index.equals("6")) {
			Algorithm1 a6 = new Algorithm1(sat);

			Optional<BitVector> rez = a6.solve(null);
			rezultat(rez, true);
			
		} else {
			throw new IllegalArgumentException();
		}
				 
	}

}
