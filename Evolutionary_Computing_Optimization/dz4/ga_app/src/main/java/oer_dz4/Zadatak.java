package oer_dz4;

public class Zadatak {
	public static void main(String[] args) {
		String datoteka = args[0];
		String[] funkcija = new String[0];
		
		GeneticAlgorithm ga = new GeneticAlgorithm(datoteka, funkcija);
		
		ga.startAlgo();
		
	}
}
