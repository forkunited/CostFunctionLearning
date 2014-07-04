package cost.model.factoredcost;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.util.Pair;
import ark.util.SerializationUtil;

/**
 * FactoredCost is an abstract parent to classes that
 * compute factored cost vectors (referred to as 's' in 
 * paper/nips2014.pdf) for cost learning models.
 * Given a model's prediction for
 * a datum, the FactoredCost outputs a vector that
 * represents the incorrect prediction sets into which
 * that prediction falls.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> label type
 */
public abstract class FactoredCost<D extends Datum<L>, L> {
	/**
	 * Initialize the factored cost for some model and data 
	 * 
	 * @param model
	 * @param data
	 * @return true if everything went okay, false otherwise.
	 */
	public abstract boolean init(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data);
	
	/**
	 * Computes a vector ('s' in paper/nips2014.pdf) that represents the 
	 * incorrect prediction classes for a given prediction.
	 * 
	 * @param datum
	 * @param prediction
	 * @return a sparse mapping from vector indices to values.  The values
	 * of the vector are usually binary indicators of whether the prediction
	 * is in an incorrect prediction class.
	 */
	public abstract Map<Integer, Double> computeVector(D datum, L prediction);
	
	/**
	 * @return a vector of incorrect prediction normalization constants (the
	 * constant n vector described in paper/nips2014.pdf)
	 */
	public abstract double[] getNorms();
	
	/**
	 * @return a name for the factored cost type to use in the experiment 
	 * configuration files.
	 */
	public abstract String getGenericName();
	
	/**
	 * @return length of the factored cost vectors returned by computeVector.
	 */
	public abstract int getVocabularySize();
	
	/**
	 * @param predictions
	 * @return \Kappa difficulties as described in section 2 of 
	 * papers/previous-approaches.pdf. This is mainly useful for the 
	 * alternating minimization approach described in previous-approaches.pdf,
	 * and doesn't need to be implemented unless you plan on using that 
	 * approach.
	 */
	public abstract Map<Integer, Double> computeKappas(Map<D, L> predictions);

	/**
	 * 
	 * @param index
	 * @return a name for the incorrect prediction class corresponding to 
	 * an indexed position within the vector returned by computeVector
	 */
	protected abstract String getVocabularyTerm(int index); 
	
	/**
	 * @return names of parameters that can be set through experiment configuration
	 * files.  This method and the other 'parameter' related methods are used
	 * for serialization/deserialization of experiments involving cost learning
	 * models with factored costs.
	 */
	public abstract String[] getParameterNames();
	
	/**
	 * @param parameter
	 * @return the value of a parameter with the given parameter name
	 */
	public abstract String getParameterValue(String parameter);
	
	/**
	 * Sets the value of a parameter
	 * 
	 * @param parameter
	 * @param parameterValue
	 * @param datumTools
	 * @return true if the parameter was successfully set, false otherwise
	 */
	public abstract boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools);
	
	/**
	 * @return an instantiation of a particular the FactoredCost class
	 */
	protected abstract FactoredCost<D, L> makeInstance();
	
	/*
	 * All methods below are for deserializing and instantiating factored costs from
	 * configuration files, similarly to the way in which features and models 
	 * from the ARKWater project are deserialized and instantiated.  See the ark.data.feature.Feature
	 * and ark.model.SupervisedModel classes for more detail on how this works.
	 */
	
	public Map<Integer, String> getSpecificShortNamesForIndices(Iterable<Integer> indices) {
		String prefix = getSpecificShortNamePrefix();
		Map<Integer, String> specificShortNames = new HashMap<Integer, String>();
		for (Integer index : indices) {
			specificShortNames.put(index, prefix + getVocabularyTerm(index));
		}
		
		return specificShortNames;
	}
	
	public List<String> getSpecificShortNames() {
		String prefix = getSpecificShortNamePrefix();
		int vocabularySize = getVocabularySize();
		List<String> specificShortNames = new ArrayList<String>(vocabularySize);
		for (int i = 0; i < vocabularySize; i++) {
			String vocabularyTerm = getVocabularyTerm(i);
			specificShortNames.add(prefix + ((vocabularyTerm == null) ? "" : vocabularyTerm));
		}
		
		return specificShortNames;
	}
	
	public FactoredCost<D, L> clone(Datum.Tools<D, L> datumTools) {
		return clone(datumTools, null);
	}
	
	public FactoredCost<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		FactoredCost<D, L> clone = makeInstance();
		String[] parameterNames = getParameterNames();
		for (int i = 0; i < parameterNames.length; i++) {
			String parameterValue = getParameterValue(parameterNames[i]);
			if (environment != null && parameterValue != null) {
				for (Entry<String, String> entry : environment.entrySet())
					parameterValue = parameterValue.replace("${" + entry.getKey() + "}", entry.getValue());
			}
			clone.setParameterValue(parameterNames[i], parameterValue, datumTools);
		}
		return clone;
	}

	public boolean deserialize(BufferedReader reader, boolean readGenericName, Datum.Tools<D, L> datumTools) throws IOException {
		if (readGenericName && SerializationUtil.deserializeGenericName(reader) == null)
			return false;
		
		Map<String, String> parameters = SerializationUtil.deserializeArguments(reader);
		if (parameters != null)
			for (Entry<String, String> entry : parameters.entrySet())
				this.setParameterValue(entry.getKey(), entry.getValue(), datumTools);

		return true;
	}
	
	public boolean serialize(Writer writer) throws IOException {
		int vocabularySize = getVocabularySize();
		writer.write(toString(false));
		writer.write("\t");
		
		for (int i = 0; i < vocabularySize; i++) {
			String vocabularyTerm = getVocabularyTerm(i);
			if (vocabularyTerm == null)
				continue;
			Pair<String, Integer> v = new Pair<String, Integer>(vocabularyTerm, i);
			if (!SerializationUtil.serializeAssignment(v, writer))
				return false;
			if (i != vocabularySize - 1)
				writer.write(",");
		}
		
		return true;
	}
	
	public String toString(boolean withVocabulary) {
		if (withVocabulary) {
			StringWriter stringWriter = new StringWriter();
			try {
				if (serialize(stringWriter))
					return stringWriter.toString();
				else
					return null;
			} catch (IOException e) {
				return null;
			}
		} else {
			String genericName = getGenericName();
			Map<String, String> parameters = new HashMap<String, String>();
			String[] parameterNames = getParameterNames();
			for (int i = 0; i < parameterNames.length; i++)
				parameters.put(parameterNames[i], getParameterValue(parameterNames[i]));
			StringWriter parametersWriter = new StringWriter();
			
			try {
				SerializationUtil.serializeArguments(parameters, parametersWriter);
			} catch (IOException e) {
				return null;
			}
			
			String parametersStr = parametersWriter.toString();
			return genericName + "(" + parametersStr + ")";
		}
	}
	
	public String toString() {
		return toString(false);
	}
	
	
	public boolean fromString(String str, Datum.Tools<D, L> datumTools) {
		try {
			return deserialize(new BufferedReader(new StringReader(str)), true, datumTools);
		} catch (IOException e) {
			
		}
		return true;
	}
	
	protected String getSpecificShortNamePrefix() {
		StringBuilder shortNamePrefixBuilder = new StringBuilder();
		String genericName = shortenName(getGenericName());
		String[] parameterNames = getParameterNames();
		
		shortNamePrefixBuilder = shortNamePrefixBuilder.append(genericName).append("_");
		for (int i = 0; i < parameterNames.length; i++)
			shortNamePrefixBuilder = shortNamePrefixBuilder.append(shortenName(parameterNames[i]))
														.append("-")
														.append(getParameterValue(parameterNames[i]))
														.append("_");
		
		return shortNamePrefixBuilder.toString();
	}
	
	private String shortenName(String name) {
		if (name.length() == 0)
			return name;
		
		StringBuilder shortenedName = new StringBuilder();
		shortenedName.append(name.charAt(0));
		
		int curWordSize = 0;
		for (int i = 1; i < name.length(); i++) {
			if (Character.isUpperCase(name.charAt(i))) {
				shortenedName.append(name.charAt(i));
				curWordSize = 1;
			} else if (curWordSize < 4) {
				shortenedName.append(name.charAt(i));
				curWordSize++;
			}
		}
		
		return shortenedName.toString();
	}
	
}
