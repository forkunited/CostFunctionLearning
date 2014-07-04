package cost.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;

/**
 * SupervisedModelCLSVMAlt is an implementation of a 
 * cost learning SVM that minimizes objective function (1) 
 * that uses the alternating
 * minimization approach from section 2 of the
 * paper/previous-approaches.pdf document.  Specifically,
 * this implementation alternates minimizing objective (1) with
 * minimizing a dot product with K-difficulty as described in 
 * section 2.  This implementation is deprecated as it extends from
 * the deprecated cost.model.SupervisedModelCL.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> label type
 * 
 * @deprecated  There is currently 
 * no recently implemented cost
 * learning SVM that minimizes objective function (1) with the
 * alternating minimization approach from section 2
 * paper/previous-approaches.pdf.  If you want one, then you should
 * implement it by extending ark.model.SupervisedModelSVM. 
 *
 */
public class SupervisedModelCLSVMAlt<D extends Datum<L>, L> extends SupervisedModelCLSVM<D, L> {
	private CostWeightComparator costWeightComparator;
	
	private class CostWeightComparator implements Comparator<Integer> {
	    @Override
	    public int compare(Integer i1, Integer i2) {
	    	double u_1 = cost_v[i1];
	    	double u_2 = cost_v[i2];
	    	
	    	if (u_1 > u_2)
	    		return -1;
	    	else if (u_1 < u_2)
	    		return 1;
	    	else 
	    		return 0;
	    }
	}
	
	
	@Override
	protected boolean initializeTraining(FeaturizedDataSet<D, L> data) {	
		if (!super.initializeTraining(data))
			return false;
			
		this.costWeightComparator = new CostWeightComparator();
		
		return true;
	}
	
	@Override
	protected boolean trainOneIteration(FeaturizedDataSet<D, L> data) {
		for (D datum : data) {	
			L datumLabel = this.mapValidLabel(datum.getLabel());
			L bestLabel = argMaxScoreLabel(data, datum, true);
			boolean datumLabelBest = datumLabel.equals(bestLabel);
			
			Map<Integer, Double> datumFeatureValues = data.getFeatureVocabularyValues(datum);
			
			if (iteration == 0) {
				List<Integer> missingNameKeys = new ArrayList<Integer>();
				for (Integer key : datumFeatureValues.keySet())
					if (!this.featureNames.containsKey(key))
						missingNameKeys.add(key);
				this.featureNames.putAll(data.getFeatureVocabularyNamesForIndices(missingNameKeys));
			}
			
			// Update feature weights
			for (int i = 0; i < this.feature_w.length; i++) {
				if (this.l1 == 0 && this.feature_w[i] == 0 && datumLabelBest)
					continue;
				
				feature_g[i] = this.l2*this.feature_w[i]-labelFeatureValue(data, datumFeatureValues, i, datumLabel)+labelFeatureValue(data, datumFeatureValues, i, bestLabel);
				
				this.feature_G[i] += feature_g[i]*feature_g[i];
				this.feature_u[i] += feature_g[i];
				
				if (this.feature_G[i] == 0)
					continue;
				if (this.l1 == 0)
					this.feature_w[i] -= feature_g[i]*this.n/Math.sqrt(this.feature_G[i]); 
				else {
					if (Math.abs(this.feature_u[i])/this.t <= this.l1)
						this.feature_w[i] = 0; 
					else 
						this.feature_w[i] = -Math.signum(this.feature_u[i])*this.n*(this.t/(Math.sqrt(this.feature_G[i])))*((Math.abs(this.feature_u[i])/this.t)-this.l1); 
				}
			}
			
			if (datumLabelBest) {
				this.t++;
				continue;
			}
			
			// Update label biases
			for (int i = 0; i < this.bias_b.length; i++) {
				bias_g[i] = ((this.labelIndices.get(datumLabel) == i) ? -1.0 : 0.0) +
								(this.labelIndices.get(bestLabel) == i ? 1.0 : 0.0);
				
				this.bias_G[i] += bias_g[i]*bias_g[i];
				this.bias_u[i] += bias_g[i];
				if (this.bias_G[i] == 0)
					continue;
				this.bias_b[i] -= bias_g[i]*this.n/Math.sqrt(this.bias_G[i]);
			}
			
			this.t++;
		}

		if (!trainCostWeights(data))
			return false;
		
		this.iteration++;

		return true;
	}
	
	private boolean trainCostWeights(FeaturizedDataSet<D, L> data) {
		Map<D, L> predictions = new HashMap<D, L>();
		for (D datum : data) {
			predictions.put(datum, argMaxScoreLabel(data, datum, true));
		}
		
		Map<Integer, Double> kappas = this.factoredCost.computeKappas(predictions);
		for (int i = 0; i < this.cost_v.length; i++) {
			if (!kappas.containsKey(i))
				this.cost_v[i] = 0;
			else
				this.cost_v[i] = -kappas.get(i);
		}
		
		// Project cost weights onto simplex \sum v_i = 1, v_i >= 0
		// Find p = max { j : u_j - (1/j)((\sum^j u_i) - 1.0) > 0 } 
		// where u is sorted desc
		Arrays.sort(this.cost_i, this.costWeightComparator);
		double sumV = 0;
		double theta = 0;
		for (int p = 0; p < this.cost_v.length; p++) {
			sumV += this.cost_v[this.cost_i[p]];
			double prevTheta = theta;
			theta = (sumV-1.0)/p;
			if (this.cost_v[this.cost_i[p]]-theta <= 0) {
				theta = prevTheta;
				break;
			}
		}
		
		for (int j = 0; j < this.cost_v.length; j++) {
			this.cost_v[j] = Math.max(0, this.cost_v[j]-theta);
		}
		
		return true;
	}
	
	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelCLSVMAlt<D, L>();
	}
	
	@Override
	public String getGenericName() {
		return "CLSVMAlt";
	}
	
	public SupervisedModel<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		SupervisedModelCLSVMAlt<D, L> clone = (SupervisedModelCLSVMAlt<D, L>)super.clone(datumTools, environment);
		
		clone.labelIndices = this.labelIndices;
		clone.trainingIterations = this.trainingIterations;
		if (this.factoredCost != null) {
			clone.factoredCost = this.factoredCost.clone(datumTools, environment);
		}
		
		return clone;
	}
}
