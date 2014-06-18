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

public abstract class FactoredCost<D extends Datum<L>, L> {
	public abstract boolean init(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data);
	public abstract Map<Integer, Double> computeVector(D datum, L prediction);
	public abstract double[] getNorms();
	public abstract String getGenericName();
	public abstract int getVocabularySize();
	
	public abstract Map<Integer, Double> computeKappas(Map<D, L> predictions);

	protected abstract String getVocabularyTerm(int index); 
	
	public abstract String[] getParameterNames();
	public abstract String getParameterValue(String parameter);
	public abstract boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools);
	protected abstract FactoredCost<D, L> makeInstance();
	
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
