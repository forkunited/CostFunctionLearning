package cost.model.factoredcost;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;

public class FactoredCostLabel<D extends Datum<L>, L> extends FactoredCost<D, L> {
	public enum FactorMode {
		ACTUAL,
		PREDICTED
	}
	
	public enum Norm {
		NONE,
		SOME
	}
	
	private String[] parameterNames = { "c", "factorMode", "norm" };
	private double c;
	private FactorMode factorMode;
	private Norm norm;
	
	private SupervisedModel<D, L> model;
	private List<L> labels;
	private double[] norms;
	
	public FactoredCostLabel() {
		this.labels = new ArrayList<L>();
		this.factorMode = FactorMode.ACTUAL;
		this.norm = Norm.SOME;
		this.norms = new double[0];
	}
	
	@Override
	public Map<Integer, Double> computeVector(D datum, L prediction) {
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		L actual = this.model.mapValidLabel(datum.getLabel());
		if (prediction.equals(actual))
			return vector;
		
		for (int i = 0; i < this.labels.size(); i++) {
			if ((this.factorMode.equals(FactorMode.PREDICTED) && this.labels.get(i).equals(prediction))
					|| (this.factorMode.equals(FactorMode.ACTUAL) && this.labels.get(i).equals(actual))) {
				vector.put(i, this.c);
			}
		}
		
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
		else if (parameter.equals("factorMode"))
			return this.factorMode.toString();
		else if (parameter.equals("norm"))
			return String.valueOf(this.norm);
		else
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("c"))
			this.c = Double.valueOf(parameterValue);
		else if (parameter.equals("factorMode"))
			this.factorMode = FactorMode.valueOf(parameterValue);
		else if (parameter.equals("norm"))
			this.norm = Norm.valueOf(parameterValue);
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
		
		if (this.norm == Norm.SOME) {
			for (int i = 0; i < vocabularySize; i++) {
				int actualIndex = i / (this.labels.size() - 1);
				L actualLabel = this.labels.get(actualIndex);
				double actualCount = dist.containsKey(actualLabel) ? dist.get(actualLabel) : 0;
				this.norms[i] = actualCount;
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
		return "Label";
	}

	@Override
	public int getVocabularySize() {
		return this.labels.size();
	}

	@Override
	protected String getVocabularyTerm(int index) {
		return this.labels.get(index).toString();
	}
	
	@Override
	protected FactoredCost<D, L> makeInstance() {
		return new FactoredCostLabel<D, L>();
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
