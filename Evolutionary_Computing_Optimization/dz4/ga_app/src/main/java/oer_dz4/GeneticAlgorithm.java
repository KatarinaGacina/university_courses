package oer_dz4;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Scanner;
import java.util.stream.Collectors;

public class GeneticAlgorithm {
	private String configFile = "config.txt";
	
	private int numVariables;
	private List<List<Double>> X;
	private Double[] y;
	private double yP;
	
	private String[] F = new String[]{"+", "-", "*", "/", "sin", "cos", "sqrt", "log", "exp"};
	private int choice = 3;
	private double constMin = -3;
	private double constMax = 3;
	
	private int initialMaxTreeDepth = 6;
	private int maxTreeDepth = 7;
	private int maxNumNodes = 30;
	
	private int popSize = 500;
	private List<Node> populacija;
	
	private int costEval = 0;
	private int maxCostEval = 1000000;
	
	private int tournirSize = 7;
	private int linearScaling = 1;
	
	private int elitism = 1;
	
	private double mutProb = 0.14;
	private double crossProb = 0.85;
	
	private double precision = 1e-9;
	
	public void setParameters() {
		
		if (this.configFile != null) {
			Properties properties = new Properties();
			
	        try (FileReader reader = new FileReader("files/config.txt")) {
	            properties.load(reader);
	            
	            String pF = properties.getProperty("FunctionNodes");
	            if (pF != null) {
	            	this.F = pF.split("\\s*,\\s*");
	            }
	            
	            String c = properties.getProperty("ConstantRange");
	            if (c != null) {
		            if (c.equals("N/A")) {
		            	this.choice = 2;
		            } else {
		            	this.choice = 3;
		            	this.constMin = Double.parseDouble(c.split("\\s*,\\s*")[0]);
		            	this.constMax = Double.parseDouble(c.split("\\s*,\\s*")[1]);
		            }
	            }
	            
	            String pom;
	            
	            pom = properties.getProperty("PopulationSize");
	            if (pom != null) {
	            	this.popSize = Integer.parseInt(pom);
	            }
	            
	            pom = properties.getProperty("TournamentSize");
	            if (pom != null) {
	            	this.tournirSize = Integer.parseInt(pom);
	            }
	            
	            boolean crosSet = false;
	            pom = properties.getProperty("CostEvaluations");
	            if (pom != null) {
	            	this.maxCostEval = Integer.parseInt(pom);
	            	crosSet = true;
	            }
	            
	            boolean mutSet = false;
	            pom = properties.getProperty("MutationProbability");
	            if (pom != null) {
	            	this.mutProb = Double.parseDouble(pom);
	            	mutSet = true;
	            }
	            
	            if (crosSet == false && mutSet == true) {
	            	this.crossProb = 1.0 - this.mutProb;
	            } else if (crosSet == true && mutSet == false) {
	            	this.mutProb = 1.0 - this.crossProb;
	            }
	            
	            pom = properties.getProperty("CrossoverProbability");
	            if (pom != null) {
	            	this.crossProb = Double.parseDouble(pom);
	            }
	            
	            pom = properties.getProperty("Precision");
	            if (pom != null) {
	            	this.precision = Double.parseDouble(pom);
	            }
	            
	            pom = properties.getProperty("MaxTreeDepth");
	            if (pom != null) {
	            	this.maxTreeDepth = Integer.parseInt(pom);
	            }
	            
	            pom = properties.getProperty("MaxNodesNumber");
	            if (pom != null) {
	            	this.maxNumNodes = Integer.parseInt(pom);
	            }
	            
	            pom = properties.getProperty("InitialMaxTreeDepth");
	            if (pom != null) {
	            	this.initialMaxTreeDepth = Integer.parseInt(pom);
	            }
	            
	            pom = properties.getProperty("UseLinearScaling");
	            if (pom != null) {
	            	this.linearScaling = Integer.parseInt(pom);
	            }

	        } catch (IOException e) {
	            e.printStackTrace();
	        }
		}
		
		this.populacija = new ArrayList<>();
	}
	
	public void setData(String datoteka) {
		try {
            File file = new File(datoteka);

            Scanner scanner = new Scanner(file);

            this.X = new ArrayList<>();
            List<Double> yList = new ArrayList<>();
            
            double sumaY = 0;
            
            int prviProlaz = 0;
            while (scanner.hasNext()) {
                String[] values = scanner.nextLine().split("\\s+");
                if (prviProlaz == 0) {
                	this.numVariables = values.length - 1;
                	prviProlaz++;
                }
                
                List<Double> xRow = new ArrayList<>();
                for (int i = 0; i < values.length - 1; i++) {
                    xRow.add(Double.parseDouble(values[i]));
                }
                this.X.add(xRow);
                
                double yi = Double.parseDouble(values[values.length - 1]);
                sumaY += yi;
                yList.add(yi);
            }
            	
            this.y = yList.toArray(new Double[0]);
            this.yP = sumaY / (double)this.y.length;
            
            scanner.close();
            
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
		
	}
	
	public GeneticAlgorithm() {
	}
	
	public GeneticAlgorithm(String datoteka, String[] zadatak) {
		this.setData(datoteka);
		this.setParameters();
	}	
	
	private double costFunction(Node jedinka) {
		this.costEval++;
		
		if (linearScaling != 0) {
			double sumaT = 0;
			
			List<Double> t_i = new ArrayList<>();
			for (int i = 0; i < this.y.length; i++) {
				double t = jedinka.evaluate(X.get(i));
				t_i.add(t);
			
				sumaT += t;
			}
			double tP = sumaT / (double)t_i.size();
			
			double sumab = 0;
			double suman = 0;
			for (int i = 0; i < this.y.length; i++) {
				sumab += ((t_i.get(i) - tP) * (this.y[i] - this.yP));
				suman += Math.pow((t_i.get(i) - tP), 2);
			}
			
			double b = sumab / suman;
			double a = tP - b * this.yP;
			
			double suma = 0;
			for (int i = 0; i < t_i.size(); i++) {
				double t = (a + b * t_i.get(i));
				t = Math.round(t / precision) * precision;
				suma += Math.pow((y[i] - t), 2);
			} 
			
			return suma / (double)(this.y.length);
			
		} else {
			double suma = 0;
			double t_i;
			for (int i = 0; i < this.y.length; i++) {
				t_i = jedinka.evaluate(X.get(i));
				t_i = Math.round(t_i / precision) * precision;
				suma += Math.pow((y[i] - t_i), 2);
			}
			
			return suma / (double)(this.y.length);
		}
	}
	
	private void full(int depth, Node node) {	
		if (depth == 1) {
			if (node instanceof BinaryNode) {
				((BinaryNode) node).setLeftChild(createTerminalNode(node));
				((BinaryNode) node).setRightChild(createTerminalNode(node));
				
			} else if (node instanceof UnaryNode) {
				((UnaryNode) node).setChild(createTerminalNode(node));
			}
            
        } else {
        	if (node instanceof BinaryNode) {
				((BinaryNode) node).setLeftChild(createNonTerminalNode(node));
				full(depth - 1, ((BinaryNode) node).getLeftChild());
				((BinaryNode) node).setRightChild(createNonTerminalNode(node));
				full(depth - 1, ((BinaryNode) node).getRightChild());
				
			} else if (node instanceof UnaryNode) {
				((UnaryNode) node).setChild(createNonTerminalNode(node));
				full(depth - 1, ((UnaryNode) node).getChild());
			}
        }
		
		return;
	}
	
	private Node createTerminalNode(Node parent) {

		Random rand = new Random();
		int c = rand.nextInt(this.choice - 1);		
		
		if (c == 1) {
			double k = this.constMin + (this.constMax - this.constMin * rand.nextDouble());
			return new ConstLeaf(k, parent);
			
		} else {
			int index = rand.nextInt(this.numVariables);
			return new Leaf(index, parent);
		}
	}	
	
	private Node createNonTerminalNode(Node parent) {
		
		Random rand = new Random();
        int i = rand.nextInt(this.F.length);
        
        IBinaryOperator ib = Operators.getBinaryOperator(this.F[i]); 
        if (ib != null) {
        	return new BinaryNode(ib, parent);
        	
        } else {
        	 IUnaryOperator in = Operators.getUnaryOperator(this.F[i]);
        	 
        	 if (in == null) {
        		 throw new IllegalArgumentException("Unknown operator!");
        	 }
        	 
        	 return new UnaryNode(in, parent);
        }

	}
	
	private void grow(int depth, Node node) {
		if (depth == 1) {
			if (node instanceof BinaryNode) {
				((BinaryNode) node).setLeftChild(createTerminalNode(node));
				((BinaryNode) node).setRightChild(createTerminalNode(node));
				
			} else if (node instanceof UnaryNode) {
				((UnaryNode) node).setChild(createTerminalNode(node));
			}
            
        } else {
        	Random rand = new Random();
        	
        	if (node instanceof BinaryNode) {
            	int index = rand.nextInt(this.choice);
        		
        		if (index == 0) {
        			((BinaryNode) node).setLeftChild(createNonTerminalNode(node));
        			
    				grow(depth - 1, ((BinaryNode) node).getLeftChild());
        			
        		} else if (index == 1) {
        			int i = rand.nextInt(this.numVariables);
        			((BinaryNode) node).setLeftChild(new Leaf(i, node));
        		
        		} else {
        			double k = this.constMin + (this.constMax - this.constMin * rand.nextDouble());
        			((BinaryNode) node).setLeftChild(new ConstLeaf(k, node));
        		}
        		
        		index = rand.nextInt(this.choice);
        		
        		if (index == 0) {
        			((BinaryNode) node).setRightChild(createNonTerminalNode(node));
        			
    				grow(depth - 1, ((BinaryNode) node).getRightChild());
        			
        		} else if (index == 1) {
        			int i = rand.nextInt(this.numVariables);
        			((BinaryNode) node).setRightChild(new Leaf(i, node));
        		
        		} else {
        			double k = this.constMin + (this.constMax - this.constMin * rand.nextDouble());
        			((BinaryNode) node).setRightChild(new ConstLeaf(k, node));
        		}
				
			} else if (node instanceof UnaryNode) {
				int index = rand.nextInt(this.choice);
				
				if (index == 0) {
        			((UnaryNode) node).setChild(createNonTerminalNode(node));
        			
    				grow(depth - 1, ((UnaryNode) node).getChild());
        			
        		} else if (index == 1) {
        			int i = rand.nextInt(this.numVariables);
        			((UnaryNode) node).setChild(new Leaf(i, node));
        		
        		} else {
        			double k = this.constMin + (this.constMax - this.constMin * rand.nextDouble());
        			((UnaryNode) node).setChild(new ConstLeaf(k, node));
        		}
			}
        }
		
		return;
	}

	private void createInitialPopulation() {
		
		if (this.initialMaxTreeDepth == 1) {
			while (this.populacija.size() < this.popSize) {
				this.populacija.add(this.createTerminalNode(null));
			}
			
		} else {
			int v = this.popSize / (this.initialMaxTreeDepth - 1);
			
			for (int d = 2; d < this.initialMaxTreeDepth; d++) {
				for (int i = 0; i < (v / 2); i++) {
					Node root = createNonTerminalNode(null);
					this.full(d, root);
					if (root.countNodes() <= this.maxNumNodes) {
						this.populacija.add(root);
					}
				}
				
				for (int i = 0; i < (v / 2); i++) {
					Node root = createNonTerminalNode(null);
					this.grow(d, root);
					if (root.countNodes() <= this.maxNumNodes) {
						this.populacija.add(root);
					}
				}
			}
			
			if (this.populacija.size() < this.popSize) {
				int n = 0;
				while (this.populacija.size() < this.popSize) {
					if (n % 2 == 0) {
						Node root = createNonTerminalNode(null);
						this.full(this.initialMaxTreeDepth - 1, root);
						if (root.countNodes() <= this.maxNumNodes) {
							this.populacija.add(root);
						}
						
					} else {
						Node root = createNonTerminalNode(null);
						this.grow(this.initialMaxTreeDepth - 1, root);
						if (root.countNodes() <= this.maxNumNodes) {
							this.populacija.add(root);
						}
					}
				}
			}
		}
	}

	private Node kTournament() {
		 List<Node> turnir = new ArrayList<>();
		 
		 Random rand = new Random();
	     for (int i = 0; i < this.tournirSize; i++) {
	    	 int index = rand.nextInt(this.popSize);
	         turnir.add(this.populacija.get(index));
	     }

	    Collections.sort(turnir, (n1, n2) -> Double.compare(costFunction(n1), costFunction(n2)));
		
		return (turnir.get(0).copyTree(null));
	}
	
	private Node createNewSubtree(int depth) {
		Random rand = new Random();
		int index = rand.nextInt(this.choice);
		
		if (index == 0) {
			Node root;
			do {
				root = createNonTerminalNode(null);
				this.grow(depth, root);
			} while(root.countNodes() > this.maxNumNodes);
			
			return root;
			
		} else if (index == 1) {
			int i = rand.nextInt(this.numVariables);
			return new Leaf(i, null);
			
		} else {
			double k = this.constMin + (this.constMax - this.constMin * rand.nextDouble());
			return new ConstLeaf(k, null);
		}

	}
	
	private Node mutacija(Node node) {
		Random rand = new Random();
		int n = node.countNodes();
		int nr = rand.nextInt(n);
		
		Node odabraniCvor = node.getByIndex(nr);
		Node parent = odabraniCvor.getParent();

		int newdepth = rand.nextInt(this.maxTreeDepth - (node.treeDepth() - odabraniCvor.treeDepth()));
		Node novi = createNewSubtree(newdepth);
		novi.setParent(parent);
		
		if (parent != null) {
		    if (parent instanceof BinaryNode) {
		    	if (((BinaryNode) parent).getLeftChild() == odabraniCvor) {
		    		((BinaryNode) parent).setLeftChild(novi);
		    		
		    	} else {
		    		((BinaryNode) parent).setRightChild(novi);
		    	}
		    } else if (parent instanceof UnaryNode) {
		        ((UnaryNode) parent).setChild(novi);
		    }
		}
		
		if (node.printExpression().equals(novi.printExpression()) || node.treeDepth() > this.maxTreeDepth || node.countNodes() > this.maxNumNodes) {
			return null;
		}
		
		return node;
	}
	
	private List<Node> krizanje(Node node1, Node node2) {
		Random rand = new Random();
		
		int n1 = node1.countNodes();
		int nr1 = rand.nextInt(n1);
		
		int n2 = node2.countNodes();
		int nr2 = rand.nextInt(n2);
		
		Node sNode1 = node1.getByIndex(nr1);
		Node sNode2 = node2.getByIndex(nr2);
		
		Node pom = sNode1.getParent();
		sNode1.setParent(sNode2.getParent());
		
		Node parent1 = sNode1.getParent();
		if (parent1 != null) {
		    if (parent1 instanceof BinaryNode) {
		    	if (((BinaryNode) parent1).getLeftChild() == sNode2) {
		    		((BinaryNode) parent1).setLeftChild(sNode1);
		    		
		    	} else {
		    		((BinaryNode) parent1).setRightChild(sNode1);
		    	}
		    } else if (parent1 instanceof UnaryNode) {
		        ((UnaryNode) parent1).setChild(sNode1);
		    }
		}
		
		sNode2.setParent(pom);
		Node parent2 = sNode2.getParent();
		if (parent2 != null) {
		    if (parent2 instanceof BinaryNode) {
		    	if (((BinaryNode) parent2).getLeftChild() == sNode1) {
		    		((BinaryNode) parent2).setLeftChild(sNode2);
		    		
		    	} else {
		    		((BinaryNode) parent2).setRightChild(sNode2);
		    	}
		    } else if (parent2 instanceof UnaryNode) {
		        ((UnaryNode) parent2).setChild(sNode2);
		    }
		}
		
		List<Node> djeca = new ArrayList<Node>();
		
		if (node1.printExpression().equals(node2.printExpression())) {
			return djeca;
		}
		
		int depth1 = node1.treeDepth();
		int depth2 = node2.treeDepth();
		int numNode1 = node1.countNodes();
		int numNode2 = node2.countNodes();
 		if (depth1 <= this.maxTreeDepth && depth2 <= this.maxTreeDepth && numNode1 <= this.maxNumNodes && numNode2 <= this.maxNumNodes) {
 			djeca.add(node1);
 			djeca.add(node2);
 		} else if (depth1 <= this.maxTreeDepth && numNode1 <= this.maxNumNodes) {
 			djeca.add(node1);
 		} else if (depth2 <= this.maxTreeDepth && numNode2 <= this.maxNumNodes) {
 			djeca.add(node2);
 		}
				
		return djeca;
	}
	
	private List<Node> elitizam() {
		Map<Node, Double> evalRj = new HashMap<>();
		for (int i = 0; i < this.popSize; i++) {
			evalRj.put(this.populacija.get(i), costFunction(this.populacija.get(i))); 
			
		}
		
		List<Node> topSolutions = evalRj.entrySet().stream()
			       .sorted(Map.Entry.comparingByValue(Comparator.naturalOrder())) //.reverseOrder()
			       .limit(this.elitism)
			       .map(Map.Entry::getKey)
			       .collect(Collectors.toList());
		
		return topSolutions;
	}
	
	/*private Node elitizam2() {
		int topRj = 0;
		double fitNajbolje = fitFunction(this.populacija[0]);
		for (int i = 1; i < this.popSize; i++) {
			double fitTrenutne = fitFunction(this.populacija[i]); 
			if (fitTrenutne < fitNajbolje) {
				fitNajbolje = fitTrenutne;
				topRj = i;
			}
		}
		
		return this.populacija[topRj];
	}*/
	
	public void startAlgo() {
		this.costEval = 0;
		
		createInitialPopulation();
		
		/*for (Node n : this.populacija) {
			System.out.println(n.treeDepth());
		}
		System.out.println();*/
		
		Node bestSolution = null;
		double minCost = -1;
		
		List<Node> novaPop;
		while (costEval < maxCostEval) { //dodatan uvjet: stagnira li rjesenje?
			novaPop = new ArrayList<>();
			
			List<Node> topSolutions = elitizam();
			for (Node t : topSolutions) {
				novaPop.add(t);
			}
			//novaPop.add(elitizam2());
			
			if (bestSolution == null || (costFunction(bestSolution) > costFunction(topSolutions.get(0)))) {
				bestSolution = topSolutions.get(0).copyTree(null);
				minCost = this.costFunction(bestSolution);
				
				System.out.println(bestSolution.printExpression());
				System.out.println(minCost);
				System.out.println();
				
				if (minCost <= this.precision) {
					break;
				}
				
				this.costEval--;
				
			}

			while (novaPop.size() < this.popSize) {
				
				Random rand = new Random();
				double p = rand.nextDouble();
				
				if (p < this.crossProb) {
					Node node1 = kTournament();
					Node node2 = kTournament();
					
					List<Node> crossNodes = krizanje(node1, node2);
					for (Node c : crossNodes) {
						novaPop.add(c);
					}
					
				} else if (p < (this.crossProb + this.mutProb)) {
					Node node = kTournament();
					
					Node mNode = mutacija(node);
					if (mNode != null) {
						novaPop.add(mNode);
					}
					
				} else {
					Node node = kTournament();
					
					novaPop.add(node);
				}
				
 			}
			
			this.populacija = novaPop;
			
			/*System.out.println();
			for (Node n : this.populacija) {
				System.out.println(n.treeDepth());
			}
			System.out.println();*/
		}
		
	}
	
}
