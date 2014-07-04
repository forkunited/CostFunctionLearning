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

/**
 * FactoredCostLabelPair factors a cost
 * function by sets of incorrect predictions (represented
 * in paper/nips2014.pdf as the vector labeled 's'), where each
 * set contains predictions of a a single actual/predicted label
 * pair (these are in the set denoted "S^o" in paper/nips2014.pdf).
 * 
 * See cost.model.factoredcost.FactoredCost for generic documentation
 * on FactoredCosts including this one.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> label type
 * 
 */
public class FactoredCostLabelPair<D extends Datum<L>, L> extends FactoredCost<D, L> {
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
	
	public FactoredCostLabelPair() {
		this.labels = new ArrayList<L>();
		this.norms = new double[0];
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
		int n = this.labels.size();
		vector.put(actualIndex*(n-1)+((predictedIndex > actualIndex) ? predictedIndex-1 : predictedIndex), this.c);
		
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
				int actualIndex = i / (this.labels.size() - 1);
				int rowPosition = i % (this.labels.size() - 1);
				int predictedIndex = rowPosition < actualIndex ? rowPosition : rowPosition + 1;
				
				L actualLabel = this.labels.get(actualIndex);
				L predictedLabel = this.labels.get(predictedIndex);
				double actualCount = dist.containsKey(actualLabel) ? dist.get(actualLabel) : 0;
				double predictedCount = dist.containsKey(predictedLabel) ? dist.get(predictedLabel) : 0;
				
				this.norms[i] = actualCount*predictedCount/N;
			}
		} else if (this.norm == Norm.LOGICAL) {
			for (int i = 0; i < vocabularySize; i++) {
				int actualIndex = i / (this.labels.size() - 1);
				L actualLabel = this.labels.get(actualIndex);
				double actualCount = dist.containsKey(actualLabel) ? dist.get(actualLabel) : 0;
				this.norms[i] = actualCount;
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
				int actualIndex = i / (this.labels.size() - 1);
				int rowPosition = i % (this.labels.size() - 1);
				int predictedIndex = rowPosition < actualIndex ? rowPosition : rowPosition + 1;
				L actual = this.labels.get(actualIndex);
				L predicted = this.labels.get(predictedIndex);
				int count = 0;
				if (actualPredictedCounts.containsKey(actual) && actualPredictedCounts.get(actual).containsKey(predicted))
					count += actualPredictedCounts.get(actual).get(predicted);
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
		return "LabelPair";
	}

	@Override
	public int getVocabularySize() {
		return this.labels.size() * (this.labels.size() - 1);
	}

	@Override
	protected String getVocabularyTerm(int index) {
		int n = this.labels.size();
		int actualIndex = index / (n - 1);
		int rowPosition = index % (n - 1);
		int predictedIndex = rowPosition < actualIndex ? rowPosition : rowPosition + 1;
		
		return "A_" + this.labels.get(actualIndex).toString() + "P_" + this.labels.get(predictedIndex).toString();
	}
	
	@Override
	protected FactoredCost<D, L> makeInstance() {
		return new FactoredCostLabelPair<D, L>();
	}

	@Override
	public Map<Integer, Double> computeKappas(Map<D, L> predictions) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public double[] getNorms() {
		return this.norms;
	}
}
