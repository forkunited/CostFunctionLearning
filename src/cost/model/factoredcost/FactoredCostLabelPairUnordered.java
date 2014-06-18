package cost.model.factoredcost;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.util.FileUtil;
import ark.util.Pair;

public class FactoredCostLabelPairUnordered<D extends Datum<L>, L> extends FactoredCost<D, L> {
	public enum Norm {
		NONE,
		LOGICAL,
		EXPECTED,
		MODEL
	}
	
	
	private String[] parameterNames = { "c", "norm", "modelPath", "modelType", "modelName" };
	private double c;
	private Norm norm = Norm.NONE;
	private String modelPath;
	private String modelType;
	private String modelName;
	
	private SupervisedModel<D, L> model;
	private List<L> labels;
	private double[] norms;
	
	public FactoredCostLabelPairUnordered() {
		this.labels = new ArrayList<L>();
		this.norms =  new double[0];
		this.modelPath = "";
		this.modelName = "";
		this.modelType = "";
	}
	
	@Override
	public Map<Integer, Double> computeVector(D datum, L prediction) {
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		L actual = this.model.mapValidLabel(datum.getLabel());
		if (prediction.equals(actual) || actual == null || prediction == null)
			return vector;
		
		int actualIndex = this.labels.indexOf(actual);
		int predictedIndex = this.labels.indexOf(prediction);
		
		int rowIndex = (actualIndex < predictedIndex) ? predictedIndex : actualIndex;
		int columnIndex = (actualIndex < predictedIndex) ? actualIndex : predictedIndex;
		vector.put(rowIndex*(rowIndex-1)/2+columnIndex, this.c);
		
		return vector;
	}

	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public String getParameterValue(String parameter) {
		if (parameter.equals("c"))
			return String.valueOf(this.c);
		else if (parameter.equals("norm"))
			return this.norm.toString();
		else if (parameter.equals("modelPath"))
			return this.modelPath;
		else if (parameter.equals("modelType"))
			return this.modelType;
		else if (parameter.equals("modelName"))
			return this.modelName;
		else
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("c"))
			this.c = Double.valueOf(parameterValue);
		else if (parameter.equals("norm"))
			this.norm = Norm.valueOf(parameterValue);
		else if (parameter.equals("modelPath"))
			this.modelPath = parameterValue;
		else if (parameter.equals("modelType"))
			this.modelType = parameterValue;
		else if (parameter.equals("modelName"))
			this.modelName = parameterValue;
		else
			return false;
		return true;
	}

	@Override
	public boolean init(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data) {
		this.model = model;
		this.labels = new ArrayList<L>();
		this.labels.addAll(this.model.getValidLabels());
		
		int N = data.size();
		int vocabularySize = getVocabularySize();
		this.norms = new double[vocabularySize];
		Map<L, Integer> dist = new HashMap<L, Integer>();
		for (D datum : data) {
			L label = model.mapValidLabel(datum.getLabel());
			if (!dist.containsKey(label))
				dist.put(label, 0);
			dist.put(label, dist.get(label) + 1);
		}
		
		if (this.norm == Norm.EXPECTED) {	
			for (int i = 0; i < vocabularySize; i++) {
				int labelIndex1 = (int)Math.floor(0.5*(Math.sqrt(8*i+1)+1));
				int labelIndex2 = i - labelIndex1*(labelIndex1-1)/2;
				L label1 = this.labels.get(labelIndex1);
				L label2 = this.labels.get(labelIndex2);
				double labelCount1 = dist.containsKey(label1) ? dist.get(label1) : 0;
				double labelCount2 = dist.containsKey(label2) ? dist.get(label2) : 0;
				
				this.norms[i] = 2.0*labelCount1*labelCount2/N;
			}
		} else if (this.norm == Norm.LOGICAL) {
			for (int i = 0; i < vocabularySize; i++) {
				int labelIndex1 = (int)Math.floor(0.5*(Math.sqrt(8*i+1)+1));
				int labelIndex2 = i - labelIndex1*(labelIndex1-1)/2;
				L label1 = this.labels.get(labelIndex1);
				L label2 = this.labels.get(labelIndex2);
				double labelCount1 = dist.containsKey(label1) ? dist.get(label1) : 0;
				double labelCount2 = dist.containsKey(label2) ? dist.get(label2) : 0;
				
				this.norms[i] = labelCount1 + labelCount2;
			}
		} else if (this.norm == Norm.MODEL) {
			SupervisedModel<D, L> normModel = data.getDatumTools().makeModelInstance(this.modelType);
			File modelFile = new File(data.getDatumTools().getDataTools().getPath(this.modelPath).getValue(), this.modelName);
			BufferedReader reader = FileUtil.getFileReader(modelFile.getAbsolutePath());
			try {
				normModel.deserialize(reader, true, true, data.getDatumTools(), "");
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			Map<D, L> predictions = normModel.classify(data);
			Map<L, Map<L, Integer>> actualPredictedCounts = new HashMap<L, Map<L, Integer>>();
			for (Entry<D, L> entry : predictions.entrySet()) {
				L actualLabel = model.mapValidLabel(entry.getKey().getLabel());
				L predictedLabel = model.mapValidLabel(entry.getValue());
				if (!actualPredictedCounts.containsKey(actualLabel))
					actualPredictedCounts.put(actualLabel, new HashMap<L, Integer>());
				if (!actualPredictedCounts.get(actualLabel).containsKey(predictedLabel))
					actualPredictedCounts.get(actualLabel).put(predictedLabel, 0);
				actualPredictedCounts.get(actualLabel).put(predictedLabel, actualPredictedCounts.get(actualLabel).get(predictedLabel) + 1);		
			}
			
			for (int i = 0; i < vocabularySize; i++) {
				int labelIndex1 = (int)Math.floor(0.5*(Math.sqrt(8*i+1)+1));
				int labelIndex2 = i - labelIndex1*(labelIndex1-1)/2;
				L label1 = this.labels.get(labelIndex1);
				L label2 = this.labels.get(labelIndex2);
				int count = 0;
				if (actualPredictedCounts.containsKey(label1) && actualPredictedCounts.get(label1).containsKey(label2))
					count += actualPredictedCounts.get(label1).get(label2);
				if (actualPredictedCounts.containsKey(label2) && actualPredictedCounts.get(label2).containsKey(label1))
					count += actualPredictedCounts.get(label2).get(label1);		
				this.norms[i] = count;
			}
		} else { 
			for (int i = 0; i < vocabularySize; i++) {
				this.norms[i] = N;
			}
		}

		return true;
	}
	
	@Override
	public String getGenericName() {
		return "LabelPairUnordered";
	}

	@Override
	public int getVocabularySize() {
		return this.labels.size() * (this.labels.size() - 1)/2;
	}

	@Override
	protected String getVocabularyTerm(int index) {
		int rowIndex = (int)Math.floor(0.5*(Math.sqrt(8*index+1)+1));
		int columnIndex = index - rowIndex*(rowIndex-1)/2;
		return this.labels.get(rowIndex).toString() + "_" + this.labels.get(columnIndex).toString();
	}
	
	@Override
	protected FactoredCost<D, L> makeInstance() {
		return new FactoredCostLabelPairUnordered<D, L>();
	}

	public List<Pair<L, L>> getUnorderedLabelPairs() {
		int vocabularySize = getVocabularySize();
		List<Pair<L, L>> unorderedLabelPairs = new ArrayList<Pair<L, L>>();
		for (int i = 0; i < vocabularySize; i++) {
			int rowIndex = (int)Math.floor(0.5*(Math.sqrt(8*i+1)+1));
			int columnIndex = i - rowIndex*(rowIndex-1)/2;
			unorderedLabelPairs.add(new Pair<L, L>(this.labels.get(rowIndex), this.labels.get(columnIndex)));
		}
		
		return unorderedLabelPairs;
	}
	
	@Override
	public Map<Integer, Double> computeKappas(Map<D, L> predictions) {
		Map<Integer, Double> kappas = new HashMap<Integer, Double>();
		Map<Integer, Double> actual = new HashMap<Integer, Double>(); // Actual p(cost_S > 0)
		Map<L, Double> labelActualP = new HashMap<L, Double>();
		Map<L, Double> labelPredictedP = new HashMap<L, Double>();
		double normalizedIncrement = 1.0/predictions.size();
		for (Entry<D, L> entry : predictions.entrySet()) {
			Map<Integer, Double> actualValues = computeVector(entry.getKey(), entry.getValue());
			for (Entry<Integer, Double> actualValue : actualValues.entrySet()) {
				if (actualValue.getValue() == 0)
					continue;
				if (!actual.containsKey(actualValue.getKey()))
					actual.put(actualValue.getKey(), 0.0);
				actual.put(actualValue.getKey(), actual.get(actualValue.getKey()) + normalizedIncrement);
			}
			
			if (!labelActualP.containsKey(entry.getKey().getLabel()))
				labelActualP.put(entry.getKey().getLabel(), 0.0);
			labelActualP.put(entry.getKey().getLabel(), labelActualP.get(entry.getKey().getLabel()) + normalizedIncrement);
			
			if (!labelPredictedP.containsKey(entry.getValue()))
				labelPredictedP.put(entry.getValue(), 0.0);
			labelPredictedP.put(entry.getValue(), labelPredictedP.get(entry.getValue()) + normalizedIncrement);
		}
		
		int vocabularySize = getVocabularySize();
		for (int i = 0; i < vocabularySize; i++) {
			double actualValue = 0.0;
			double expectedValue = 0.0;
			
			int rowIndex = (int)Math.floor(0.5*(Math.sqrt(8*i+1)+1));
			int columnIndex = i - rowIndex*(rowIndex-1)/2;
			L l_1 = this.labels.get(rowIndex);
			L l_2 = this.labels.get(columnIndex);
			if (labelActualP.containsKey(l_1) && labelPredictedP.containsKey(l_2))
				expectedValue += labelActualP.get(l_1)*labelPredictedP.get(l_2);
			if (labelActualP.containsKey(l_2) && labelPredictedP.containsKey(l_1))
				expectedValue += labelActualP.get(l_2)*labelPredictedP.get(l_1);
			
			if (actual.containsKey(i))
				actualValue = actual.get(i);
			
			if (expectedValue == 1.0) {
				kappas.put(i, 0.0);
			} else {
				kappas.put(i, (actualValue - expectedValue)/(1.0 - expectedValue));
			}
		}
		
		return kappas;
	}
	
	@Override
	public double[] getNorms() {
		return this.norms;
	}
}
