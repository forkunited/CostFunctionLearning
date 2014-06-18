package cost.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;

public class SupervisedModelCLSVMN<D extends Datum<L>, L> extends SupervisedModelCLSVM<D, L> {
	@Override
	protected boolean initializeTraining(FeaturizedDataSet<D, L> data) {	
		//... Don't need to do anything extra here...
		
		return super.initializeTraining(data);
	}
	
	@Override
	protected boolean trainOneIteration(FeaturizedDataSet<D, L> data) {
		double N = data.size();
		double c = Double.valueOf(this.getHyperParameterValue("c"));
		for (D datum : data) {	
			L datumLabel = this.mapValidLabel(datum.getLabel());
			L bestLabel = argMaxScoreLabel(data, datum, true);
			boolean datumLabelBest = datumLabel.equals(bestLabel);
			
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
				
				this.feature_g[i] = this.l2*this.feature_w[i]/N-labelFeatureValue(data, datumFeatureValues, i, datumLabel)+labelFeatureValue(data, datumFeatureValues, i, bestLabel);
				
				this.feature_G[i] += this.feature_g[i]*this.feature_g[i];
				this.feature_u[i] += this.feature_g[i];
				
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
			
			// Update label biases
			for (int i = 0; i < this.bias_b.length; i++) {
				bias_g[i] = ((this.labelIndices.get(datumLabel) == i) ? -1.0 : 0.0) +
								(this.labelIndices.get(bestLabel) == i ? 1.0 : 0.0);
				
				this.bias_G[i] += this.bias_g[i]*this.bias_g[i];
				this.bias_u[i] += this.bias_g[i];
				if (this.bias_G[i] == 0)
					continue;
				this.bias_b[i] -= this.bias_g[i]*this.n/Math.sqrt(this.bias_G[i]);
			}
		
			// Update cost weights
			double[] costNorms = this.factoredCost.getNorms();
			Map<Integer, Double> costs = this.factoredCost.computeVector(datum, bestLabel);
			
			for (int i = 0; i < costNorms.length; i++) {
				if (costNorms[i] == 0)
					continue;
				
				double cost = (costs.containsKey(i) ? costs.get(i) : 0);
				double costNorm = costNorms[i];
				
				this.cost_g[i] = cost+c*costNorm*this.cost_v[i]/N-c*costNorm/N;
				this.cost_u[i] += this.cost_g[i];
				this.cost_G[i] += this.cost_g[i]*this.cost_g[i];
	
				if (this.cost_G[i] == 0)
					continue;
				
				this.cost_v[i] -= this.cost_g[i]*this.n/Math.sqrt(this.cost_G[i]); 
				
				if (this.cost_v[i] < 0)
					this.cost_v[i] = 0;
			}
			
			this.t++;
		}
		
		this.iteration++;

		return true;
	}
	
	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelCLSVMN<D, L>();
	}
	
	@Override
	public String getGenericName() {
		return "CLSVMN";
	}
	
	public double objectiveValue(FeaturizedDataSet<D, L> data) {
		double value = super.objectiveValue(data);
		double c = Double.valueOf(this.getHyperParameterValue("c"));
		
		double costNNorm = 0;
		double costChoices = 0;
		double[] costNorms = this.factoredCost.getNorms();
		for (int i = 0; i < this.cost_v.length; i++) {
			costNNorm += this.cost_v[i]*this.cost_v[i]*costNorms[i];
			costChoices -= this.cost_v[i]*costNorms[i];
		}
		
		costNNorm *= c/2.0;
		costChoices *= c;
		
		value += costNNorm - costChoices;
		
		return value;
	}
	
	public SupervisedModel<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		SupervisedModelCLSVMN<D, L> clone = (SupervisedModelCLSVMN<D, L>)super.clone(datumTools, environment);
		
		clone.labelIndices = this.labelIndices;
		clone.trainingIterations = this.trainingIterations;
		if (this.factoredCost != null) {
			clone.factoredCost = this.factoredCost.clone(datumTools, environment);
		}
		
		return clone;
	}
}
