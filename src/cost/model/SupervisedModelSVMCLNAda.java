package cost.model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import cost.data.annotation.CostDatumTools;
import cost.model.factoredcost.FactoredCost;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.util.Pair;
import ark.util.SerializationUtil;

/**
 * SupervisedModelSVMCLNAda is an implementation of the
 * cost learning SVM described in paper/nips2014.pdf that
 * uses Adagrad to minimize the objective function.  This
 * implementation is slow as it does full rather than sparse
 * updates of the weight vector.  The newer 
 * SupervisedModelSVMCLN is an improved implementation that
 * uses sparse updates.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> label type
 * 
 * @deprecated Use cost.model.SupervisedModelSVMCLN instead.
 *
 */
public class SupervisedModelSVMCLNAda<D extends Datum<L>, L> extends SupervisedModelSVMAda<D, L> {
	protected FactoredCost<D, L> factoredCost;
	protected double[] cost_v;
	protected double[] cost_u; 
	protected double[] cost_G;  // Just diagonal
	protected double[] cost_g;
	
	public SupervisedModelSVMCLNAda() {
		super();
	}
	
	@Override
	protected boolean deserializeExtraInfo(String name, BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		if (name.equals("factoredCost")) {
			String genericCost = SerializationUtil.deserializeGenericName(reader);
			this.factoredCost = ((CostDatumTools<D,L>)datumTools).makeFactoredCostInstance(genericCost);
			if (!this.factoredCost.deserialize(reader, false, datumTools))
				return false;
		} else {
			return super.deserializeExtraInfo(name, reader, datumTools);
		}
		
		return true;
	}

	@Override
	protected boolean serializeExtraInfo(Writer writer) throws IOException {
		if (!super.serializeExtraInfo(writer))
			return false;
		
		if (this.factoredCost != null) {
			writer.write("\t");
			Pair<String, String> factoredCostAssignment = new Pair<String, String>("factoredCost", this.factoredCost.toString(false));
			if (!SerializationUtil.serializeAssignment(factoredCostAssignment, writer))
				return false;
			writer.write("\n");
		}
		
		return true;
	}
	
	@Override
	protected boolean initializeTraining(FeaturizedDataSet<D, L> data) {
		if (!super.initializeTraining(data))
			return false;
		
		if (!this.factoredCost.init(this, data))
			return false;
		
		if (this.cost_v == null) {
			this.cost_v = new double[this.factoredCost.getVocabularySize()];
			this.cost_u = new double[this.cost_v.length];
			this.cost_G = new double[this.cost_v.length];
		}
		
		this.cost_g = new double[this.cost_v.length];
		
		return true;
	}
	
	@Override
	protected boolean trainOneDatum(D datum, L datumLabel, L bestLabel, int iteration, FeaturizedDataSet<D, L> data) {
		int N = data.size();
		
		if (!super.trainOneDatum(datum, datumLabel, bestLabel, iteration, data))
			return false;
		
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
		
		return true;
	}
	
	@Override
	public double objectiveValue(FeaturizedDataSet<D, L> data) {
		double value = super.objectiveValue(data);
		
		for (D datum : data) {
			double maxScore = maxScoreLabel(data, datum, true);
			double datumScore = scoreLabel(data, datum, datum.getLabel(), false);
			value += maxScore - datumScore;
		}
		
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
	
	protected double scoreLabel(FeaturizedDataSet<D, L> data, D datum, L label, boolean includeCost) {
		double score = super.scoreLabel(data, datum, label, false);

		if (includeCost) {
			Map<Integer, Double> costs = this.factoredCost.computeVector(datum, label);
			for (Entry<Integer, Double> entry : costs.entrySet())
				score += this.cost_v[entry.getKey()]*entry.getValue();
		}
		
		return score;
	}
	
	public SupervisedModel<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		SupervisedModelSVMCLNAda<D, L> clone = (SupervisedModelSVMCLNAda<D, L>)super.clone(datumTools, environment);
		
		clone.labelIndices = this.labelIndices;
		clone.trainingIterations = this.trainingIterations;
		if (this.factoredCost != null) {
			clone.factoredCost = this.factoredCost.clone(datumTools, environment);
		}
		
		return clone;
	}
	
	@Override
	protected boolean deserializeParameters(BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		Pair<String, String> tAssign = SerializationUtil.deserializeAssignment(reader);
		Pair<String, String> numWeightsAssign = SerializationUtil.deserializeAssignment(reader);
		Pair<String, String> numCostsAssign = SerializationUtil.deserializeAssignment(reader);
		
		int numWeights = Integer.valueOf(numWeightsAssign.getSecond());
		int numCosts = Integer.valueOf(numCostsAssign.getSecond());
		int numFeatures = numWeights / this.labelIndices.size();
		
		this.t = Integer.valueOf(tAssign.getSecond());
		this.featureNames = new HashMap<Integer, String>();
		
		this.feature_w = new HashMap<Integer, Double>();
		this.feature_u = new HashMap<Integer, Double>();
		this.feature_G = new HashMap<Integer, Double>();
		
		this.bias_b = new double[this.labelIndices.size()];
		this.bias_u = new double[this.bias_b.length];
		this.bias_G = new double[this.bias_b.length];
		
		this.cost_v = new double[numCosts];
		this.cost_u = new double[this.cost_v.length];
		this.cost_G = new double[this.cost_v.length];
	
		String assignmentLeft = null;
		while ((assignmentLeft = SerializationUtil.deserializeAssignmentLeft(reader)) != null) {
			if (assignmentLeft.equals("labelFeature")) {
				String labelFeature = SerializationUtil.deserializeGenericName(reader);
				Map<String, String> featureParameters = SerializationUtil.deserializeArguments(reader);
				
				String featureName = labelFeature.substring(labelFeature.indexOf("-") + 1);
				double w = Double.valueOf(featureParameters.get("w"));
				double G = Double.valueOf(featureParameters.get("G"));
				double u = Double.valueOf(featureParameters.get("u"));
				int labelIndex = Integer.valueOf(featureParameters.get("labelIndex"));
				int featureIndex = Integer.valueOf(featureParameters.get("featureIndex"));
				
				int index = labelIndex*numFeatures+featureIndex;
				this.featureNames.put(featureIndex, featureName);
				
				if (w != 0)
					this.feature_w.put(index, w);
				
				this.feature_u.put(index, u);
				this.feature_G.put(index, G);
			} else if (assignmentLeft.equals("labelBias")) {
				SerializationUtil.deserializeGenericName(reader);
				Map<String, String> biasParameters = SerializationUtil.deserializeArguments(reader);
				double b = Double.valueOf(biasParameters.get("b"));
				double G = Double.valueOf(biasParameters.get("G"));
				double u = Double.valueOf(biasParameters.get("u"));
				int index = Integer.valueOf(biasParameters.get("index"));
				
				this.bias_b[index] = b;
				this.bias_G[index] = G;
				this.bias_u[index] = u;
			} else if (assignmentLeft.equals("cost")) {
				SerializationUtil.deserializeGenericName(reader);
				Map<String, String> costParameters = SerializationUtil.deserializeArguments(reader);
				double v = Double.valueOf(costParameters.get("v"));
				double G = Double.valueOf(costParameters.get("G"));
				double u = Double.valueOf(costParameters.get("u"));
				int index = Integer.valueOf(costParameters.get("index"));
				
				this.cost_v[index] = v;
				this.cost_G[index] = G;
				this.cost_u[index] = u;
			} else {
				break;
			}
		}
		
		return true;
	}
	
	@Override
	protected boolean serializeParameters(Writer writer) throws IOException {
		Pair<String, String> tAssignment = new Pair<String, String>("t", String.valueOf(this.t));
		if (!SerializationUtil.serializeAssignment(tAssignment, writer))
			return false;
		writer.write("\n");
		
		Pair<String, String> numFeatureWeightsAssignment = new Pair<String, String>("numWeights", String.valueOf(this.labelIndices.size()*this.numFeatures));
		if (!SerializationUtil.serializeAssignment(numFeatureWeightsAssignment, writer))
			return false;
		writer.write("\n");
		
		Pair<String, String> numCostWeightsAssignment = new Pair<String, String>("numCosts", String.valueOf(this.cost_v.length));
		if (!SerializationUtil.serializeAssignment(numCostWeightsAssignment, writer))
			return false;
		writer.write("\n");
		
		for (int i = 0; i < this.labelIndices.size(); i++) {
			String label = this.labelIndices.reverseGet(i).toString();
			String biasValue = label +
					  "(b=" + this.bias_b[i] +
					  ", G=" + this.bias_G[i] +
					  ", u=" + this.bias_u[i] +
					  ", index=" + i +
					  ")";

			Pair<String, String> biasAssignment = new Pair<String, String>("labelBias", biasValue);
			if (!SerializationUtil.serializeAssignment(biasAssignment, writer))
				return false;
			writer.write("\n");
		}
		
		for (int i = 0; i < this.labelIndices.size(); i++) {
			String label = this.labelIndices.reverseGet(i).toString();
			for (Entry<Integer, String> featureName : this.featureNames.entrySet()) {
				int weightIndex = getWeightIndex(i, featureName.getKey());
				double w = (this.feature_w.containsKey(weightIndex)) ? this.feature_w.get(weightIndex) : 0;
				double G = (this.feature_G.containsKey(weightIndex)) ? this.feature_G.get(weightIndex) : 0;
				double u = (this.feature_u.containsKey(weightIndex)) ? this.feature_u.get(weightIndex) : 0;
				
				if (w == 0 && G == 0 && u == 0)
					continue;
				
				String featureValue = label + "-" + 
									  featureName.getValue() + 
									  "(w=" + w +
									  ", G=" + G +
									  ", u=" + u +
									  ", labelIndex=" + i +
									  ", featureIndex=" + featureName.getKey() + 
									  ")";
				
				Pair<String, String> featureAssignment = new Pair<String, String>("labelFeature", featureValue);
				if (!SerializationUtil.serializeAssignment(featureAssignment, writer))
					return false;
				writer.write("\n");
			}
		}
		
		if (this.factoredCost != null) {
			List<String> costNames = this.factoredCost.getSpecificShortNames();
			for (int i = 0; i < costNames.size(); i++) {
				String costValue = costNames.get(i) +
								   "(v=" + this.cost_v[i] +
								   ", G=" + this.cost_G[i] +
								   ", u=" + this.cost_u[i] +
								   ", index=" + i +
								   ")";
				
				Pair<String, String> costAssignment = new Pair<String, String>("cost", costValue);
				if (!SerializationUtil.serializeAssignment(costAssignment, writer))
					return false;
				writer.write("\n");
			}
		}

		writer.write("\n");
		
		return true;
	}
	
	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelSVMCLNAda<D, L>();
	}

	@Override
	public String getGenericName() {
		return "SVMCLN";
	}
	
	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		if (this.factoredCost != null && !this.factoredCost.init(this, data))
			return null;

		return super.posterior(data);
	}
}
