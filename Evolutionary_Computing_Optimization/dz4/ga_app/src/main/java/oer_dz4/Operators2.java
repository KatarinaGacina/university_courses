package oer_dz4;

import java.util.HashMap;
import java.util.Map;

public class Operators2 {
	
	static Map<IOperator, String> operatorsMap = new HashMap<IOperator, String>() {
		private static final long serialVersionUID = 1L;

	{
	    put(plus, "+");
	    put(minus, "-");
	    put(puta, "*");
	    put(dijeljeno, "/");
	    put(log, "log");
	    put(sin, "sin");
	    put(cos, "cos");
	    put(sqrt, "sqrt");
	    put(exp, "exp");
	}};
	
	public static String getSymbol(IOperator o) {
		return operatorsMap.get(o);
	}
	
	
	
	public static final IOperator plus = (values) -> {
		return values[0] + values[1]; 
	};
	
	public static final IOperator minus = (values) -> {
		return values[0] - values[1]; 
	};
	
	public static final IOperator puta = (values) -> {
		return values[0] * values[1]; 
	};
	
	public static final IOperator dijeljeno = (values) -> {
		if (values[1] == 0) {
			return 1.0;
		}
		
		return values[0] / values[1]; 
	};
	
	
	
	public static final IOperator log = (values) -> {
		Double rj = Math.log10(values[0]);
		if (Double.isNaN(rj)) {
			rj = 1.0;
		}
		
		return rj; 
	};
	
	public static final IOperator sin = (values) -> {
		return Math.sin(values[0]); 
	};
	
	public static final IOperator cos = (values) -> {
		return Math.cos(values[0]); 
	};
	
	public static final IOperator sqrt = (values) -> {
		if (values[0] < 0) {
			return 1.0;
		}
		
		return Math.sqrt(values[0]); 
	};
	
	public static final IOperator exp = (values) -> {
		return Math.exp(values[0]); 
	};
	
}
