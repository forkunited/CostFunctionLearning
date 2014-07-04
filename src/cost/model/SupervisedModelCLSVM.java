package cost.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;

/**
 * SupervisedModelCLSVM is an implementation of a 
 * cost learning SVM that minimizes objective function (1) from the
 * paper/previous-approaches.pdf 
 * document. This implementation is deprecated as it extends from
 * the deprecated cost.model.SupervisedModelCL. 
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> label type
 * 
 * @deprecated There is currently no recently implemented cost
 * learning SVM that minimizes objective function (1) from 
 * paper/previous-approaches.pdf.  If you want one, then you should
 * implement it by extending ark.model.SupervisedModelSVM. 
 *
 */
public class SupervisedModelCLSVM<D extends Datum<L>, L> extends SupervisedModelCL<D, L> {	
	private CostWeightComparator costWeightComparator;
	protected double[] feature_g;
	protected double[] bias_g;
	protected double[] cost_g;
	protected int iteration;
	
	private class CostWeightComparator implements Comparator<Integer> {
	    @Override
	    public int compare(Integer i1, Integer i2) {
	    	double u_1 = cost_G[i1]*(2.0*cost_v[i1]-1);
	    	double u_2 = cost_G[i2]*(2.0*cost_v[i2]-1);
	    	
	    	if (cost_G[i1] != 0 && cost_G[i2] == 0)
	    		return -1;
	    	else if (cost_G[i1] == 0 && cost_G[i2] != 0)
	    		return 1;
	    	if (u_1 > u_2)
	    		return -1;
	    	else if (u_1 < u_2)
	    		return 1;
	    	else 
	    		return 0;
	    }
	}
	
	public SupervisedModelCLSVM() {
		super();
		this.featureNames = new HashMap<Integer, String>();
	}

	@Override
	protected boolean initializeTraining(FeaturizedDataSet<D, L> data) {		
		this.costWeightComparator = new CostWeightComparator();
		this.feature_g = new double[this.feature_w.length];
		this.bias_g = new double[this.bias_b.length];
		this.cost_g = new double[this.cost_v.length];
		
		return true;
	}
	
	@Override
	protected boolean trainOneIteration(FeaturizedDataSet<D, L> data) {
		for (D datum : data) {	
			L datumLabel = this.mapValidLabel(datum.getLabel());
			L bestLabel = argMaxScoreLabel(data, datum, true);
			boolean datumLabelBest = datumLabel.equals(bestLabel);
			
			Map<Integer, Double> bestLabelCosts = this.factoredCost.computeVector(datum, bestLabel);
			Map<Integer, Double> datumFeatureValues = data.getFeatureVocabularyValues(datum);
			
			if (this.iteration == 0) {
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
				
				this.feature_g[i] = this.l2*this.feature_w[i]-labelFeatureValue(data, datumFeatureValues, i, datumLabel)+labelFeatureValue(data, datumFeatureValues, i, bestLabel);
				
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
						this.feature_w[i] = -Math.signum(this.feature_u[i])*(this.t*this.n/(Math.sqrt(this.feature_G[i])))*((Math.abs(this.feature_u[i])/this.t)-this.l1); 
				}
			}
			
			if (datumLabelBest) {
				this.t++;
				continue;
			}
			
			// Update label biases
			for (int i = 0; i < this.bias_b.length; i++) {
				this.bias_g[i] = ((this.labelIndices.get(datumLabel) == i) ? -1.0 : 0.0) +
								(this.labelIndices.get(bestLabel) == i ? 1.0 : 0.0);
				
				this.bias_G[i] += bias_g[i]*bias_g[i];
				this.bias_u[i] += bias_g[i];
				if (this.bias_G[i] == 0)
					continue;
				this.bias_b[i] -= bias_g[i]*this.n/Math.sqrt(this.bias_G[i]);
			}
			
			// Update cost weights
			for (int i = 0; i < this.cost_v.length; i++) {
				this.cost_g[i] = (bestLabelCosts.containsKey(i)) ? bestLabelCosts.get(i) : 0;
				this.cost_G[i] += cost_g[i]*cost_g[i];
				this.cost_u[i] += cost_g[i];
				
				if (this.cost_G[i] != 0)
					this.cost_v[i] -= cost_g[i]*this.n/Math.sqrt(this.cost_G[i]); 
			}
			
			// Project cost weights onto simplex \sum v_i = 1, v_i >= 0
			// Find p = max { j : u_j - 1/G_j((\sum^j u_i) - 1.0)/(\sum^j 1.0/G_i) > 0 } 
			// where u and G are sorted desc
			Arrays.sort(this.cost_i, costWeightComparator);
			double sumV = 0;
			double harmonicG = 0;
			double theta = 0;
			for (int p = 0; p < this.cost_v.length; p++) {
				if (this.cost_G[this.cost_i[p]] != 0) {
					sumV += this.cost_v[this.cost_i[p]];
					harmonicG += 1.0/this.cost_G[this.cost_i[p]];
				}
				double prevTheta = theta;
				theta = (sumV-1.0)/harmonicG;
				if (this.cost_G[this.cost_i[p]] == 0 || this.cost_v[this.cost_i[p]]-theta/this.cost_G[this.cost_i[p]] <= 0) {
					theta = prevTheta;
					break;
				}
			}
			
			for (int j = 0; j < this.cost_v.length; j++) {
				if (this.cost_G[j] == 0)
					this.cost_v[j] = 0;
				else
					this.cost_v[j] = Math.max(0, this.cost_v[j]-theta/this.cost_G[j]);
			}
			
			this.t++;
		}
		
		this.iteration++;

		return true;
	}
	
	private double maxScoreLabel(FeaturizedDataSet<D, L> data, D datum, boolean includeCost) {
		double maxScore = Double.NEGATIVE_INFINITY;
		for (L label : this.validLabels) {
			double score = scoreLabel(data, datum, label, includeCost);
			if (score >= maxScore) {
				maxScore = score;
			}
		}
		return maxScore;
	}
	
	protected L argMaxScoreLabel(FeaturizedDataSet<D, L> data, D datum, boolean includeCost) {
		L maxLabel = null;
		double maxScore = Double.NEGATIVE_INFINITY;
		for (L label : this.validLabels) {
			double score = scoreLabel(data, datum, label, includeCost);
			if (score >= maxScore) {
				maxScore = score;
				maxLabel = label;
			}
		}
		return maxLabel;
	}
	
	protected double scoreLabel(FeaturizedDataSet<D, L> data, D datum, L label, boolean includeCost) {
		double score = 0;		

		Map<Integer, Double> featureValues = data.getFeatureVocabularyValues(datum);
		int labelIndex = this.labelIndices.get(label);
		int numFeatures = data.getFeatureVocabularySize();
		int weightIndexOffset = labelIndex*numFeatures;
		for (Entry<Integer, Double> entry : featureValues.entrySet()) {
			score += this.feature_w[weightIndexOffset + entry.getKey()]*entry.getValue();
		}
		
		score += this.bias_b[labelIndex];

		if (includeCost) {
			Map<Integer, Double> costs = this.factoredCost.computeVector(datum, label);
			for (Entry<Integer, Double> entry : costs.entrySet())
				score += costs.get(entry.getKey())*this.cost_v[entry.getKey()];
		}
		
		return score;
	}
	
	protected double labelFeatureValue(FeaturizedDataSet<D,L> data, Map<Integer, Double> featureValues, int weightIndex, L label) {
		int labelIndex = this.labelIndices.get(label);
		int numFeatures = data.getFeatureVocabularySize();
		int featureLabelIndex = weightIndex / numFeatures;
		if (featureLabelIndex != labelIndex)
			return 0.0;
		
		int featureIndex = weightIndex % numFeatures;
		if (!featureValues.containsKey(featureIndex))
			return 0.0;
		else
			return featureValues.get(featureIndex);
	}
	
	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> posteriors = new HashMap<D, Map<L, Double>>(data.size());
		if (this.factoredCost != null && !this.factoredCost.init(this, data))
			return null;
		for (D datum : data) {
			posteriors.put(datum, posteriorForDatum(data, datum));
		}
		
		return posteriors;
	}

	private Map<L, Double> posteriorForDatum(FeaturizedDataSet<D, L> data, D datum) {
		Map<L, Double> posterior = new HashMap<L, Double>(this.validLabels.size());
		double[] scores = new double[this.validLabels.size()];
		double max = Double.NEGATIVE_INFINITY;
		for (L label : this.validLabels) {
			double score = scoreLabel(data, datum, label, false);
			scores[this.labelIndices.get(label)] = score;
			if (score > max)
				max = score;
		}
		
		double lse = 0;
		for (int i = 0; i < scores.length; i++)
			lse += Math.exp(scores[i] - max);
		lse = max + Math.log(lse);
		
		for (L label : this.validLabels) {
			posterior.put(label, Math.exp(scores[this.labelIndices.get(label)]-lse));
		}
		
		return posterior;
	}
	
	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelCLSVM<D, L>();
	}
	
	@Override
	public String getGenericName() {
		return "CLSVM";
	}

	@Override
	public double computeLoss(FeaturizedDataSet<D, L> data) {
		double loss = 0;
		
		for (D datum : data) {
			double maxScore = maxScoreLabel(data, datum, true);
			double datumScore = scoreLabel(data, datum, datum.getLabel(), false);
			loss += maxScore - datumScore;
		}
		
		return loss;
	}
	
	public SupervisedModel<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		SupervisedModelCLSVM<D, L> clone = (SupervisedModelCLSVM<D, L>)super.clone(datumTools, environment);
		
		clone.labelIndices = this.labelIndices;
		clone.trainingIterations = this.trainingIterations;
		if (this.factoredCost != null) {
			clone.factoredCost = this.factoredCost.clone(datumTools, environment);
		}
		
		return clone;
	}
}

