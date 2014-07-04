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
 * SupervisedModelSVMCLNDeadPegasos is an implementation of the
 * cost learning SVM described in paper/nips2014.pdf that
 * uses the Pegasos SGD described at
 * http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf
 * to minimize the objective function.  This implementation
 * is correct except that the 1/(lambda * t) learning
 * rate is used on both the feature weights and the bias
 * terms.  Putting the lambda in the learning rate for the
 * bias terms results in incorrect biases.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> label type
 * 
 * @deprecated Use cost.model.SupervisedModelSVMCLN instead.
 *
 */
public class SupervisedModelSVMCLNDeadPegasos<D extends Datum<L>, L> extends SupervisedModelSVMDeadPegasos<D, L> {
	protected FactoredCost<D, L> factoredCost;
	protected double[] cost_v;
	protected double[] cost_g;
	
	public SupervisedModelSVMCLNDeadPegasos() {
		super();
	}
	
	@Override
	protected boolean deserializeExtraInfo(String name, BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		if (name.equals("factoredCost")) {
			String genericCost = SerializationUtil.deserializeGenericName(reader);
			this.factoredCost = ((CostDatumTools<D, L>)datumTools).makeFactoredCostInstance(genericCost);
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
			double eta = 1.0/(this.l2*this.t); // Learning rate
			
			this.cost_g[i] = cost+costNorm*this.cost_v[i]/N-costNorm/N;
			this.cost_v[i] -= this.cost_g[i]*eta; 
			
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
		
		double c = Double.valueOf(this.factoredCost.getParameterValue("c"));
		
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
		SupervisedModelSVMCLNDeadPegasos<D, L> clone = (SupervisedModelSVMCLNDeadPegasos<D, L>)super.clone(datumTools, environment);
		
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
		Pair<String, String> sAssign = SerializationUtil.deserializeAssignment(reader);
		Pair<String, String> numWeightsAssign = SerializationUtil.deserializeAssignment(reader);
		Pair<String, String> numCostsAssign = SerializationUtil.deserializeAssignment(reader);
		
		int numWeights = Integer.valueOf(numWeightsAssign.getSecond());
		int numCosts = Integer.valueOf(numCostsAssign.getSecond());
		int numFeatures = numWeights / this.labelIndices.size();
		
		this.t = Integer.valueOf(tAssign.getSecond());
		this.s = Double.valueOf(sAssign.getSecond());
		this.featureNames = new HashMap<Integer, String>();
		
		this.feature_W = new double[numWeights];
		this.bias_b = new double[this.labelIndices.size()];
		this.cost_v = new double[numCosts];
	
		String assignmentLeft = null;
		while ((assignmentLeft = SerializationUtil.deserializeAssignmentLeft(reader)) != null) {
			if (assignmentLeft.equals("labelFeature")) {
				String labelFeature = SerializationUtil.deserializeGenericName(reader);
				Map<String, String> featureParameters = SerializationUtil.deserializeArguments(reader);
				
				String featureName = labelFeature.substring(labelFeature.indexOf("-") + 1);
				double W = Double.valueOf(featureParameters.get("W"));
				int labelIndex = Integer.valueOf(featureParameters.get("labelIndex"));
				int featureIndex = Integer.valueOf(featureParameters.get("featureIndex"));
				
				int index = labelIndex*numFeatures+featureIndex;
				this.featureNames.put(featureIndex, featureName);
				this.feature_W[index] = W;
			} else if (assignmentLeft.equals("labelBias")) {
				SerializationUtil.deserializeGenericName(reader);
				Map<String, String> biasParameters = SerializationUtil.deserializeArguments(reader);
				double b = Double.valueOf(biasParameters.get("b"));
				int index = Integer.valueOf(biasParameters.get("index"));
				
				this.bias_b[index] = b;
			} else if (assignmentLeft.equals("cost")) {
				SerializationUtil.deserializeGenericName(reader);
				Map<String, String> costParameters = SerializationUtil.deserializeArguments(reader);
				double v = Double.valueOf(costParameters.get("v"));
				int index = Integer.valueOf(costParameters.get("index"));
				
				this.cost_v[index] = v;
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
		
		Pair<String, String> sAssignment = new Pair<String, String>("s", String.valueOf(this.s));
		if (!SerializationUtil.serializeAssignment(sAssignment, writer))
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
				double W = this.feature_W[weightIndex];
				
				if (W == 0) // Might need to get rid of this line if want to pause training and resume
					continue;
				
				String featureValue = label + "-" + 
									  featureName.getValue() + 
									  "(W=" + W +
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
		return new SupervisedModelSVMCLNDeadPegasos<D, L>();
	}

	@Override
	public String getGenericName() {
		return "SVMCLN";
	}
}
