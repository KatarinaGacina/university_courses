package oer_dz4;

import java.util.HashMap;
import java.util.Map;

public class Operators {
	
	public static final IBinaryOperator plus = (value1, value2) -> {
		return value1 + value2; 
	};
	
	public static final IBinaryOperator minus = (value1, value2) -> {
		return value1 - value2; 
	};
	
	public static final IBinaryOperator puta = (value1, value2) -> {
		return value1 * value2; 
	};
	
	public static final IBinaryOperator dijeljeno = (value1, value2) -> {
		if (value2 == 0) {
			return 1.0;
		}
		
		return value1 / value2; 
	};
	
	
	
	public static final IUnaryOperator log = (value) -> {
		Double rj = Math.log10(value);
		if (Double.isNaN(rj)) {
			rj = 1.0;
		}
		
		return rj; 
	};
	
	public static final IUnaryOperator sin = (value) -> {
		return Math.sin(value); 
	};
	
	public static final IUnaryOperator cos = (value) -> {
		return Math.cos(value); 
	};
	
	public static final IUnaryOperator sqrt = (value) -> {
		if (value < 0) {
			return 1.0;
		}
		
		return Math.sqrt(value); 
	};
	
	public static final IUnaryOperator exp = (value) -> {
		return Math.exp(value); 
	};
	
	static Map<IBinaryOperator, String> binaryMap = new HashMap<IBinaryOperator, String>() {
		private static final long serialVersionUID = 1L;

	{
	    put(plus, "+");
	    put(minus, "-");
	    put(puta, "*");
	    put(dijeljeno, "/");
	}};
	static Map<String, IBinaryOperator> binaryMapInverse = new HashMap<String, IBinaryOperator>() {
		private static final long serialVersionUID = 1L;

	{
	    put("+", plus);
	    put("-", minus);
	    put("*", puta);
	    put("/", dijeljeno);
	}};
	
	static Map<IUnaryOperator, String> unaryMap = new HashMap<IUnaryOperator, String>() {
		private static final long serialVersionUID = 1L;
	{
	    put(log, "log");
	    put(sin, "sin");
	    put(cos, "cos");
	    put(sqrt, "sqrt");
	    put(exp, "exp");
	}};
	static Map<String, IUnaryOperator> unaryMapInverse = new HashMap<String, IUnaryOperator>() {
		private static final long serialVersionUID = 1L;
	{
	    put("log", log);
	    put("sin", sin);
	    put("cos", cos);
	    put("sqrt", sqrt);
	    put("exp", exp);
	}};
	
	public static String getSymbol(IBinaryOperator o) {
		return binaryMap.get(o);
	}
	public static String getSymbol(IUnaryOperator o) {
		return unaryMap.get(o);
	}
	
	public static IBinaryOperator getBinaryOperator(String s) {
		return binaryMapInverse.get(s);
	}
	public static IUnaryOperator getUnaryOperator(String s) {
		return unaryMapInverse.get(s);
	}
	
}
