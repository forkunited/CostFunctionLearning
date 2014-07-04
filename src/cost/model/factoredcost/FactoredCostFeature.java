package cost.model.factoredcost;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;

/**
 * FactoredCostFeature factors a cost
 * function by sets of incorrect predictions (represented
 * in paper/nips2014.pdf as the vector labeled 's'), where each
 * set consists of incorrect predictions for which a given
 * feature takes on a non-zero value.
 * 
 * See cost.model.factoredcost.FactoredCost for generic documentation
 * on FactoredCosts including this one.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> label type
 */
public class FactoredCostFeature<D extends Datum<L>, L> extends FactoredCost<D, L> {
	private String featureReference;
	private double c;
	private String[] parameterNames = { "c", "featureReference" };

	private SupervisedModel<D, L> model;
	private FeaturizedDataSet<D, L> data;
	
	public FactoredCostFeature() {
		
	}
	
	@Override
	public Map<Integer, Double> computeVector(D datum, L prediction) {
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		L actual = this.model.mapValidLabel(datum.getLabel());
		if (prediction.equals(actual))
			return vector;
		
		Feature<D, L> feature = this.data.getFeatureByReferenceName(this.featureReference);
		vector = feature.computeVector(datum);
		for (Entry<Integer, Double> entry : vector.entrySet()) {
			entry.setValue(entry.getValue() * this.c);
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
		else if (parameter.equals("featureReference"))
			return this.featureReference;
		else
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("c"))
			this.c = Double.valueOf(parameterValue);
		else if (parameter.equals("featureReference"))
			this.featureReference = parameterValue;
		else
			return false;
		return true;
	}

	@Override
	public boolean init(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data) {
		this.model = model;
		this.data = data;
		return true;
	}
	
	@Override
	public String getGenericName() {
		return "Feature";
	}

	@Override
	public int getVocabularySize() {
		return this.data.getFeatureByReferenceName(this.featureReference).getVocabularySize();
	}

	@Override
	protected String getVocabularyTerm(int index) {
		return this.data.getFeatureByReferenceName(this.featureReference).getVocabularyTerm(index);
	}
	
	@Override
	protected FactoredCost<D, L> makeInstance() {
		return new FactoredCostFeature<D, L>();
	}

	@Override
	public Map<Integer, Double> computeKappas(Map<D, L> predictions) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[] getNorms() {
		// FIXME
		return null;
	}
}
