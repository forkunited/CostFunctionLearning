package cost.model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.BidirectionalLookupTable;
import ark.util.OutputWriter;
import ark.util.Pair;
import ark.util.SerializationUtil;

// NOTE: This version slow because no "occasional" updates
public class SupervisedModelSVMAda<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	protected BidirectionalLookupTable<L, Integer> labelIndices;
	protected int trainingIterations;
	protected Map<Integer, String> featureNames;
	protected Map<Integer, Double> feature_w; // Labels x Input features
	protected int numFeatures;
	protected double[] bias_b;
	protected double[] bias_g;
	
	// Adagrad stuff
	protected int t;
	protected Map<Integer, Double> feature_u; 
	protected Map<Integer, Double> feature_G;  // Just diagonal
	protected double[] bias_u;
	protected double[] bias_G;
	
	protected double l1;
	protected double l2;
	protected double n = 1.0;
	protected double epsilon = 0;
	protected double c = 1.0;
	protected String[] hyperParameterNames = { "l2", "l1", "c", "n", "epsilon" };
	
	protected Random random;

	public SupervisedModelSVMAda() {
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
		
		return true;
	}

	@Override
	public boolean train(FeaturizedDataSet<D, L> data, FeaturizedDataSet<D, L> testData, List<SupervisedModelEvaluation<D, L>> evaluations) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();
		
		if (!initializeTraining(data))
			return false;
		
		double prevObjectiveValue = objectiveValue(data);
		Map<D, L> prevPredictions = classify(data);
		
		output.debugWriteln("Training " + getGenericName() + " for " + this.trainingIterations + " iterations...");
		
		for (int iteration = 0; iteration < this.trainingIterations; iteration++) {
			if (!trainOneIteration(iteration, data)) 
				return false;
			
			if (iteration % 5 == 0) {
				double objectiveValue = objectiveValue(data);
				double objectiveValueDiff = objectiveValue - prevObjectiveValue;
				Map<D, L> predictions = classify(data);
				int labelDifferences = countLabelDifferences(prevPredictions, predictions);
			
				output.debugWriteln("(c=" + this.c + ", l1=" + this.l1 + ", l2=" + this.l2 + ") Finished iteration " + iteration + " objective diff: " + objectiveValueDiff + " objective: " + objectiveValue + " prediction-diff: " + labelDifferences + "/" + predictions.size() + " non-zero weights: " + this.feature_w.size() + "/" + this.numFeatures*this.labelIndices.size());
			
				if (iteration > 20 && Math.abs(objectiveValueDiff) < this.epsilon) {
					output.debugWriteln("(c=" + this.c + ", l1=" + this.l1 + ", l2=" + this.l2 + ") Terminating early at iteration " + iteration);
					break;
				}
				
				prevObjectiveValue = objectiveValue;
				prevPredictions = predictions;
			} else {
				output.debugWriteln("(c=" + this.c + ", l1=" + this.l1 + ", l2=" + this.l2 + ") Finished iteration " + iteration + " non-zero weights: " + this.feature_w.size() + "/" + this.numFeatures*this.labelIndices.size());
			}
		}
		
		return true;
	}
	
	protected boolean initializeTraining(FeaturizedDataSet<D, L> data) {
		if (this.feature_w == null) {
			this.t = 1;
			
			this.bias_b = new double[this.validLabels.size()];
			this.feature_w = new HashMap<Integer, Double>();
			this.numFeatures = data.getFeatureVocabularySize();
			
			this.bias_u = new double[this.bias_b.length];
			this.bias_G = new double[this.bias_u.length];
		
			this.feature_u = new HashMap<Integer, Double>();
			this.feature_G = new HashMap<Integer, Double>();
		}
		
		this.bias_g = new double[this.bias_b.length];
		this.random = data.getDatumTools().getDataTools().makeLocalRandom();
		
		return true;
	}
	
	protected boolean trainOneIteration(int iteration, FeaturizedDataSet<D, L> data) {
		List<Integer> dataPermutation = data.constructRandomDataPermutation(this.random);
		for (Integer datumId : dataPermutation) {
			D datum = data.getDatumById(datumId);
			L datumLabel = this.mapValidLabel(datum.getLabel());
			L bestLabel = argMaxScoreLabel(data, datum, true);
			
			if (!trainOneDatum(datum, datumLabel, bestLabel, iteration, data))
				return false;
			
			this.t++;
		}
		
		return true;
	}
	
	protected boolean trainOneDatum(D datum, L datumLabel, L bestLabel, int iteration, FeaturizedDataSet<D, L> data) {
		int N = data.size();
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
		Map<Integer, Double> feature_g = new HashMap<Integer, Double>();
		if (!datumLabelBest) {
			for (Entry<Integer, Double> featureValue : datumFeatureValues.entrySet()) {
				int datumLabelWeightIndex = getWeightIndex(datumLabel, featureValue.getKey());
				int bestLabelWeightIndex = getWeightIndex(bestLabel, featureValue.getKey());
				
				if (!this.feature_w.containsKey(datumLabelWeightIndex)) {
					this.feature_w.put(datumLabelWeightIndex, 0.0);
					if (!this.feature_G.containsKey(datumLabelWeightIndex)) {
						this.feature_G.put(datumLabelWeightIndex, 0.0);
						this.feature_u.put(datumLabelWeightIndex, 0.0);
					}
				}
				
				if (!this.feature_w.containsKey(bestLabelWeightIndex)) {
					this.feature_w.put(bestLabelWeightIndex, 0.0);
					if (!this.feature_G.containsKey(bestLabelWeightIndex)) {
						this.feature_G.put(bestLabelWeightIndex, 0.0);
						this.feature_u.put(bestLabelWeightIndex, 0.0);
					}
				}
				
				feature_g.put(datumLabelWeightIndex,  -featureValue.getValue());
				feature_g.put(bestLabelWeightIndex, featureValue.getValue());
			}
		}
		
		if (this.l1 > 0) {
			for (Entry<Integer, Double> entryG : this.feature_G.entrySet()) {
				double w = (this.feature_w.containsKey(entryG.getKey()) ? this.feature_w.get(entryG.getKey()) : 0.0);
				double g =  this.l2*w/N + ((feature_g.containsKey(entryG.getKey())) ? feature_g.get(entryG.getKey()) : 0.0);
				double u = this.feature_u.get(entryG.getKey()) + g;
				double G = entryG.getValue() + g*g;

				entryG.setValue(G);
				this.feature_u.put(entryG.getKey(), u);
			
				if (Math.abs(u)/this.t <= this.l1) {
					if (this.feature_w.containsKey(entryG.getKey()))
						this.feature_w.remove(entryG.getKey());	
				} else
					this.feature_w.put(entryG.getKey(), 
							-Math.signum(u)*(this.t*this.n/(Math.sqrt(G)))*((Math.abs(u)/this.t)-this.l1));
			}
		} else {
			Set<Integer> zeroedW = new HashSet<Integer>();
			for (Entry<Integer, Double> entryW : this.feature_w.entrySet()) {
				double g = this.l2*entryW.getValue()/N + ((feature_g.containsKey(entryW.getKey())) ? feature_g.get(entryW.getKey()) : 0.0);
				double u = this.feature_u.get(entryW.getKey()) + g;
				double G = this.feature_G.get(entryW.getKey()) + g*g;
				
				this.feature_u.put(entryW.getKey(), u);
				this.feature_G.put(entryW.getKey(), G);
				
				double newW = entryW.getValue() - g*this.n/Math.sqrt(G);
				if (Math.abs(newW) <= .00001)
					zeroedW.add(entryW.getKey());
				else
					entryW.setValue(newW); 
			}
			
			for (Integer wIndex : zeroedW)
				this.feature_w.remove(wIndex);
		}
		
		if (datumLabelBest)
			return true;
		
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
	
	protected double objectiveValue(FeaturizedDataSet<D, L> data) {
		double value = 0;
		
		if (this.l1 > 0) {
			double l1Norm = 0;
			for (double w : this.feature_w.values())
				l1Norm += Math.abs(w);
			value += l1Norm*this.l1;
		}
		
		if (this.l2 > 0) {
			double l2Norm = 0;
			for (double w : this.feature_w.values())
				l2Norm += w*w;
			value += l2Norm*this.l2*.5;
		}
		
		for (D datum : data) {
			double maxScore = maxScoreLabel(data, datum, true);
			double datumScore = scoreLabel(data, datum, datum.getLabel(), false);
			value += maxScore - datumScore;
		}
		
		return value;
	}
	
	protected double maxScoreLabel(FeaturizedDataSet<D, L> data, D datum, boolean includeCost) {
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
		for (Entry<Integer, Double> entry : featureValues.entrySet()) {
			int wIndex = this.getWeightIndex(label, entry.getKey());
			if (this.feature_w.containsKey(wIndex))
				score += this.feature_w.get(wIndex)*entry.getValue();
		}
		
		score += this.bias_b[labelIndex];

		if (includeCost) {
			if (!mapValidLabel(datum.getLabel()).equals(label))
				score += this.c;
		}
		
		return score;
	}
	
	protected int getWeightIndex(L label, int featureIndex) {
		return this.labelIndices.get(label)*this.numFeatures + featureIndex;
	}
	
	protected int getWeightIndex(int labelIndex, int featureIndex) {
		return labelIndex*this.numFeatures + featureIndex;
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
			return String.valueOf(this.c);
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
		else if (parameter.equals("c"))
			this.c = Double.valueOf(parameterValue);
		else if (parameter.equals("n"))
			this.n = Double.valueOf(parameterValue);
		else if (parameter.equals("epsilon"))
			this.epsilon = Double.valueOf(parameterValue);
		else
			return false;
		return true;
	}
	
	public SupervisedModel<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		SupervisedModelSVMAda<D, L> clone = (SupervisedModelSVMAda<D, L>)super.clone(datumTools, environment);
		
		clone.labelIndices = this.labelIndices;
		clone.trainingIterations = this.trainingIterations;
		
		return clone;
	}
	
	@Override
	protected boolean deserializeParameters(BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		Pair<String, String> tAssign = SerializationUtil.deserializeAssignment(reader);
		Pair<String, String> numWeightsAssign = SerializationUtil.deserializeAssignment(reader);
	
		int numWeights = Integer.valueOf(numWeightsAssign.getSecond());
		this.numFeatures = numWeights / this.labelIndices.size();
		
		this.t = Integer.valueOf(tAssign.getSecond());
		this.featureNames = new HashMap<Integer, String>();
		
		this.feature_w = new HashMap<Integer, Double>();
		this.feature_u = new HashMap<Integer, Double>();
		this.feature_G = new HashMap<Integer, Double>();
		
		this.bias_b = new double[this.labelIndices.size()];
		this.bias_u = new double[this.bias_b.length];
		this.bias_G = new double[this.bias_b.length];
			
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

		writer.write("\n");
		
		return true;
	}

	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelSVMAda<D, L>();
	}

	@Override
	public String getGenericName() {
		return "SVM";
	}

	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> posteriors = new HashMap<D, Map<L, Double>>(data.size());

		for (D datum : data) {
			posteriors.put(datum, posteriorForDatum(data, datum));
		}
		
		return posteriors;
	}

	protected Map<L, Double> posteriorForDatum(FeaturizedDataSet<D, L> data, D datum) {
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
}
