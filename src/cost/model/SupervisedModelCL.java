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
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.BidirectionalLookupTable;
import ark.util.OutputWriter;
import ark.util.Pair;
import ark.util.SerializationUtil;

public abstract class SupervisedModelCL<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	protected BidirectionalLookupTable<L, Integer> labelIndices;
	protected FactoredCost<D, L> factoredCost;
	protected int trainingIterations;
	protected Map<Integer, String> featureNames;
	protected double[] feature_w; // Labels x Input features
	protected double[] bias_b;
	protected double[] cost_v;
	
	// Adagrad stuff
	protected int t;
	protected double[] feature_u; 
	protected double[] feature_G;  // Just diagonal
	protected double[] bias_u;
	protected double[] bias_G;
	protected double[] cost_u; 
	protected double[] cost_G;  // Just diagonal
	protected Integer[] cost_i; // Cost indices for sorting v and G
	
	protected double l1;
	protected double l2;
	protected double n = 1.0;
	protected double epsilon = 0;
	protected String[] hyperParameterNames = { "l2", "l1", "c", "n", "epsilon" };
	
	protected abstract boolean trainOneIteration(FeaturizedDataSet<D, L> data);
	protected abstract boolean initializeTraining(FeaturizedDataSet<D, L> data);
	public abstract double computeLoss(FeaturizedDataSet<D, L> data);
	
	public SupervisedModelCL() {
		this.featureNames = new HashMap<Integer, String>();
	}
	
	@Override
	protected boolean deserializeExtraInfo(String name, BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		if (this.validLabels != null && this.labelIndices == null) {
			// Might be better to do this somewhere else...?
			this.labelIndices = new BidirectionalLookupTable<L, Integer>();
			int i = 0;
			for (L label : this.validLabels) {
				this.labelIndices.put(label, i);
				i++;
			}
		}
		
		if (name.equals("trainingIterations")) {
			this.trainingIterations = Integer.valueOf(SerializationUtil.deserializeAssignmentRight(reader));
		} else if (name.equals("factoredCost")) {
			String genericCost = SerializationUtil.deserializeGenericName(reader);
			this.factoredCost = ((CostDatumTools<D,L>)datumTools).makeFactoredCostInstance(genericCost);
			if (!this.factoredCost.deserialize(reader, false, datumTools))
				return false;
		}
		
		return true;
	}

	@Override
	protected boolean serializeExtraInfo(Writer writer) throws IOException {
		writer.write("\t");
		Pair<String, String> trainingIterationsAssignment = new Pair<String, String>("trainingIterations", String.valueOf(this.trainingIterations));
		if (!SerializationUtil.serializeAssignment(trainingIterationsAssignment, writer))
			return false;
		writer.write("\n");
		
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
	public boolean train(FeaturizedDataSet<D, L> data, FeaturizedDataSet<D, L> testData, List<SupervisedModelEvaluation<D, L>> evaluations) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();
		
		if (!this.factoredCost.init(this, data))
			return false;
		
		if (this.cost_v == null) {
			this.bias_b = new double[this.validLabels.size()];
			this.feature_w = new double[data.getFeatureVocabularySize()*this.validLabels.size()];
			this.cost_v = new double[this.factoredCost.getVocabularySize()];
			
			this.t = 1;
			
			this.feature_u = new double[this.feature_w.length];
			this.feature_G = new double[this.feature_w.length];
			
			this.bias_u = new double[this.bias_b.length];
			this.bias_G = new double[this.bias_u.length];

			this.cost_u = new double[this.cost_v.length];
			this.cost_G = new double[this.cost_v.length];
			
			this.cost_i = new Integer[this.cost_v.length];
			for (int i = 0; i < this.cost_i.length; i++)
				this.cost_i[i] = i;
			
			if (!initializeTraining(data)) 
				return false;
		}
		
		double prevObjectiveValue = objectiveValue(data);
		Map<D, L> prevPredictions = classify(data);
		
		output.debugWriteln("Training " + getGenericName() + " for " + this.trainingIterations + " iterations...");
		
		for (int iteration = 0; iteration < this.trainingIterations; iteration++) {
			if (!trainOneIteration(data))
				return false;
			
			double objectiveValue = objectiveValue(data);
			double objectiveValueDiff = objectiveValue - prevObjectiveValue;
			Map<D, L> predictions = classify(data);
			int labelDifferences = countLabelDifferences(prevPredictions, predictions);
			
			double vSum = 0;
			for (int i = 0; i < this.cost_v.length; i++)
				vSum += this.cost_v[i];
			
			output.debugWriteln("(c=" + this.factoredCost.getParameterValue("c")  + ", l1=" + this.l1 + ", l2=" + this.l2 + ") Finished iteration " + iteration + " objective diff: " + objectiveValueDiff + " objective: " + objectiveValue + " prediction-diff: " + labelDifferences + "/" + predictions.size() + " v-sum: " + vSum + ").");
			
			if (iteration > 20 && Math.abs(objectiveValueDiff) < this.epsilon) {
				output.debugWriteln("(c=" + this.factoredCost.getParameterValue("c")  + ", l1=" + this.l1 + ", l2=" + this.l2 + ") Terminating early at iteration " + iteration);
				break;
			}
			
			prevObjectiveValue = objectiveValue;
			prevPredictions = predictions;
		}
		
		return true;
	}
	
	private int countLabelDifferences(Map<D, L> labels1, Map<D, L> labels2) {
		int count = 0;
		for (Entry<D, L> entry: labels1.entrySet()) {
			if (!labels2.containsKey(entry.getKey()) || !entry.getValue().equals(labels2.get(entry.getKey())))
				count++;
		}
		return count;
	}
	
	public double objectiveValue(FeaturizedDataSet<D, L> data) {
		double value = 0;
		
		if (this.l1 > 0) {
			double l1Norm = 0;
			for (int i = 0; i < this.feature_w.length; i++)
				value += Math.abs(this.feature_w[i]);
			value += l1Norm*this.l1;
		}
		
		if (this.l2 > 0) {
			double l2Norm = 0;
			for (int i = 0; i < this.feature_w.length; i++)
				value += this.feature_w[i]*this.feature_w[i];
			value += l2Norm*this.l2*.5;
		}
		
		value += computeLoss(data);
		
		return value;
	}
	
	@Override
	protected String[] getHyperParameterNames() {
		return this.hyperParameterNames;
	}

	@Override
	public String getHyperParameterValue(String parameter) {
		if (parameter.equals("l1"))
			return String.valueOf(this.l1);
		else if (parameter.equals("l2"))
			return String.valueOf(this.l2);
		else if (parameter.equals("c"))
			return (this.factoredCost == null) ? "0" : this.factoredCost.getParameterValue("c");
		else if (parameter.equals("n"))
			return String.valueOf(this.n);
		else if (parameter.equals("epsilon"))
			return String.valueOf(this.epsilon);
		return null;
	}

	@Override
	public boolean setHyperParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("l1"))
			this.l1 = Double.valueOf(parameterValue);
		else if (parameter.equals("l2"))
			this.l2 = Double.valueOf(parameterValue);
		else if (parameter.equals("c") && this.factoredCost != null)
			this.factoredCost.setParameterValue("c", parameterValue, datumTools);
		else if (parameter.equals("n"))
			this.n = Double.valueOf(parameterValue);
		else if (parameter.equals("epsilon"))
			this.epsilon = Double.valueOf(parameterValue);
		else
			return false;
		return true;
	}
	
	public SupervisedModel<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		SupervisedModelCL<D, L> clone = (SupervisedModelCL<D, L>)super.clone(datumTools, environment);
		
		clone.labelIndices = this.labelIndices;
		clone.trainingIterations = this.trainingIterations;
		if (this.factoredCost != null) {
			clone.factoredCost = this.factoredCost.clone(datumTools, environment);
		}
		
		return clone;
	}
	
	public FactoredCost<D, L> getFactoredCost() {
		return this.factoredCost;
	}
	
	public double[] getCostWeights() {
		return this.cost_v;
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
		
		this.feature_w = new double[numWeights];
		this.feature_u = new double[this.feature_w.length];
		this.feature_G = new double[this.feature_w.length];
		
		this.bias_b = new double[this.labelIndices.size()];
		this.bias_u = new double[this.bias_b.length];
		this.bias_G = new double[this.bias_b.length];
		
		this.cost_v = new double[numCosts];
		this.cost_u = new double[this.cost_v.length];
		this.cost_G = new double[this.cost_v.length];
	
		this.cost_i = new Integer[this.cost_v.length];
		for (int i = 0; i < this.cost_i.length; i++)
			this.cost_i[i] = i;
		
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
				this.feature_w[index] = w;
				this.feature_u[index] = u;
				this.feature_G[index] = G;
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
		
		Pair<String, String> numFeatureWeightsAssignment = new Pair<String, String>("numWeights", String.valueOf(this.feature_w.length));
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
				int index = i*this.feature_w.length/this.labelIndices.size()+featureName.getKey();
				
				String featureValue = label + "-" + 
									  featureName.getValue() + 
									  "(w=" + this.feature_w[index] +
									  ", G=" + this.feature_G[index] +
									  ", u=" + this.feature_u[index] +
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
}

