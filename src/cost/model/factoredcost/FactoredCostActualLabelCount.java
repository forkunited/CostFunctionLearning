package cost.model.factoredcost;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;

public class FactoredCostActualLabelCount<D extends Datum<L>, L> extends FactoredCost<D, L> {	
	private String[] parameterNames = { "c" };
	private double c;
	
	private SupervisedModel<D, L> model;
	private List<L> labels;
	private double[] norms;
	private int[] labelCountThreshholds;
	private Map<L, Integer> labelsToIndices;
	
	public FactoredCostActualLabelCount() {
		this.labels = new ArrayList<L>();
		this.norms = new double[0];
		this.labelCountThreshholds = new int[]{ 0, 10, 20, 40, 80, 160 };
		this.labelsToIndices = new HashMap<L, Integer>();
	}
	
	@Override
	public Map<Integer, Double> computeVector(D datum, L prediction) {
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		L actual = this.model.mapValidLabel(datum.getLabel());
		if (prediction.equals(actual) || actual == null || prediction == null)
			return vector;
		
		int actualIndex = this.labelsToIndices.get(actual);
		vector.put(actualIndex, this.c);
		
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
		else
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("c"))
			this.c = Double.valueOf(parameterValue);
		else
			return false;
		return true;
	}

	@Override
	public boolean init(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data) {
		this.model = model;
		this.labels = new ArrayList<L>();
		this.labels.addAll(this.model.getValidLabels());
		this.norms = new double[this.labelCountThreshholds.length];
		
		Map<L, Integer> labelCounts = new HashMap<L, Integer>();
		for (D datum : data) {
			L actualLabel = this.model.mapValidLabel(datum.getLabel());
			if (labelCounts.containsKey(actualLabel))
				labelCounts.put(actualLabel, labelCounts.get(actualLabel) + 1);
			else
				labelCounts.put(actualLabel, 1);
		}

		for (Entry<L, Integer> entry : labelCounts.entrySet()) {
			for (int i = 1; i < this.labelCountThreshholds.length; i++) {
				this.labelsToIndices.put(entry.getKey(), this.labelCountThreshholds.length - 1);
				this.norms[this.labelCountThreshholds.length - 1] += entry.getValue();
				if (entry.getValue() < this.labelCountThreshholds[i]) {
					this.labelsToIndices.put(entry.getKey(), i-1);
					this.norms[this.labelCountThreshholds.length - 1] -= entry.getValue();
					this.norms[i-1] += entry.getValue();
					break;
				}
			}
		}
		
		return true;
	}
	
	@Override
	public String getGenericName() {
		return "ActualLabelCount";
	}

	@Override
	public int getVocabularySize() {
		return this.labelCountThreshholds.length;
	}

	@Override
	protected String getVocabularyTerm(int index) {
		return "T_" + this.labelCountThreshholds[index];
			}
	
	@Override
	protected FactoredCost<D, L> makeInstance() {
		return new FactoredCostActualLabelCount<D, L>();
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
