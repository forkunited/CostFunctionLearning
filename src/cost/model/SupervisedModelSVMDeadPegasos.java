package cost.model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.BidirectionalLookupTable;
import ark.util.OutputWriter;
import ark.util.Pair;
import ark.util.SerializationUtil;

// NOTE: This version uses pegasos algorithm. Learning rate screws up bias term.  Gah.
public class SupervisedModelSVMDeadPegasos<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	protected BidirectionalLookupTable<L, Integer> labelIndices;
	protected int trainingIterations;
	protected Map<Integer, String> featureNames;
	protected int numFeatures;
	protected double[] bias_b;
	protected double[] bias_g;

	protected int t;
	protected double[] feature_W; // Labels x Input features (scale by s to get actual weights)
	protected double s;
	
	protected double l2;
	protected double epsilon = 0;
	protected String[] hyperParameterNames = { "l2", "epsilon" };
	
	protected Random random;

	public SupervisedModelSVMDeadPegasos() {
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
		
		//double prevObjectiveValue = objectiveValue(data);
		Map<D, L> prevPredictions = classify(testData);
		List<Double> prevEvaluationValues = new ArrayList<Double>();
		for (SupervisedModelEvaluation<D, L> evaluation : evaluations) {
			prevEvaluationValues.add(evaluation.evaluate(this, testData, prevPredictions));
		}
		
		output.debugWriteln("Training " + getGenericName() + " for " + this.trainingIterations + " iterations...");
		
		for (int iteration = 0; iteration < this.trainingIterations; iteration++) {
			if (!trainOneIteration(iteration, data)) 
				return false;
			
			if (iteration % 10 == 0) {
				//double objectiveValue = objectiveValue(data);
				//double objectiveValueDiff = objectiveValue - prevObjectiveValue;
				Map<D, L> predictions = classify(testData);
				int labelDifferences = countLabelDifferences(prevPredictions, predictions);
			
				List<Double> evaluationValues = new ArrayList<Double>();
				for (SupervisedModelEvaluation<D, L> evaluation : evaluations) {
					evaluationValues.add(evaluation.evaluate(this, testData, predictions));
				}
				
				String statusStr = "(l2=" + this.l2 + ") Finished iteration " + iteration + /*" objective diff: " + objectiveValueDiff + " objective: " + objectiveValue + */" prediction-diff: " + labelDifferences + "/" + predictions.size() + " ";
				for (int i = 0; i < evaluations.size(); i++) {
					String evaluationName = evaluations.get(i).toString(false);
					double evaluationDiff = evaluationValues.get(i) - prevEvaluationValues.get(i);
					statusStr += evaluationName + " diff: " + evaluationDiff + " " + evaluationName + ": " + evaluationValues.get(i) + " ";
				}
					
				output.debugWriteln(statusStr);
				
				/*
				if (iteration > 20 && Math.abs(objectiveValueDiff) < this.epsilon) {
					output.debugWriteln("(l2=" + this.l2 + ") Terminating early at iteration " + iteration);
					break;
				}*/
				
				// prevObjectiveValue = objectiveValue;
				prevPredictions = predictions;
				prevEvaluationValues = evaluationValues;
			} else {
				output.debugWriteln("(l2=" + this.l2 + ") Finished iteration " + iteration);
			}
		}
		
		return true;
	}
	
	protected boolean initializeTraining(FeaturizedDataSet<D, L> data) {
		if (this.feature_W == null) {
			this.t = 1;
			this.s = 1;
			
			this.bias_b = new double[this.validLabels.size()];
			this.numFeatures = data.getFeatureVocabularySize();
			this.feature_W = new double[this.validLabels.size()*this.numFeatures];	
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

			if (!trainOneDatum(datum, datumLabel, bestLabel, iteration, data)) {
				return false;
			}
			
			this.t++;
		}
		return true;
	}
	
	protected boolean trainOneDatum(D datum, L datumLabel, L bestLabel, int iteration, FeaturizedDataSet<D, L> data) {
		boolean datumLabelBest = datumLabel.equals(bestLabel);
		
		Map<Integer, Double> datumFeatureValues = data.getFeatureVocabularyValues(datum);
		
		if (iteration == 0) {
			List<Integer> missingNameKeys = new ArrayList<Integer>();
			for (Integer key : datumFeatureValues.keySet())
				if (!this.featureNames.containsKey(key))
					missingNameKeys.add(key);
			this.featureNames.putAll(data.getFeatureVocabularyNamesForIndices(missingNameKeys));
		}
		
		double eta = 1.0/(this.l2*this.t); // Learning rate
		this.s = (this.t > 1) ? (1.0-eta*this.l2)*this.s : 1; // Weight scalar
		
		// Update feature weights
		if (!datumLabelBest) {
			for (Entry<Integer, Double> featureValue : datumFeatureValues.entrySet()) {
				int datumLabelWeightIndex = getWeightIndex(datumLabel, featureValue.getKey());
				int bestLabelWeightIndex = getWeightIndex(bestLabel, featureValue.getKey());
				double delta = eta*featureValue.getValue()/this.s;
				this.feature_W[datumLabelWeightIndex] += delta;
				this.feature_W[bestLabelWeightIndex] -= delta;
				
			}
		}
		
		if (datumLabelBest)
			return true;
		
		// Update label biases
		for (int i = 0; i < this.bias_b.length; i++) {
			this.bias_g[i] = ((this.labelIndices.get(datumLabel) == i) ? -1.0 : 0.0) +
							(this.labelIndices.get(bestLabel) == i ? 1.0 : 0.0);
			
			this.bias_b[i] -= eta*bias_g[i];
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
		
		if (this.l2 > 0) {
			double l2Norm = 0;
			for (double W : this.feature_W)
				l2Norm += W*W*this.s*this.s;
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
		double maxScore = Double.NEGATIVE_INFINITY;
		List<L> maxLabels = null; // for breaking ties randomly
		L maxLabel = null;
		for (L label : this.validLabels) {
			double score = scoreLabel(data, datum, label, includeCost);

			if (score == maxScore) {
				if (maxLabels == null) {
					maxLabels = new ArrayList<L>();
					if (maxLabel != null) {
						maxLabels.add(maxLabel);
						maxLabel = null;
					}
				}
				maxLabels.add(label);
			} else if (score > maxScore) {
				maxScore = score;
				maxLabel = label;
				maxLabels = null;
			}
		}
		
		if (maxLabels != null)
			return maxLabels.get(this.random.nextInt(maxLabels.size()));
		else
			return maxLabel;
	}
	
	protected double scoreLabel(FeaturizedDataSet<D, L> data, D datum, L label, boolean includeCost) {
		double score = 0;		
		
		Map<Integer, Double> featureValues = data.getFeatureVocabularyValues(datum);
		int labelIndex = this.labelIndices.get(label);
		for (Entry<Integer, Double> entry : featureValues.entrySet()) {
			int wIndex = this.getWeightIndex(label, entry.getKey());
			score += this.s*this.feature_W[wIndex]*entry.getValue();
		}
		
		score += this.bias_b[labelIndex];

		if (includeCost) {
			if (!mapValidLabel(datum.getLabel()).equals(label))
				score += 1.0;
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
		if (parameter.equals("l2"))
			return String.valueOf(this.l2);
		else if (parameter.equals("epsilon"))
			return String.valueOf(this.epsilon);
		return null;
	}

	@Override
	public boolean setHyperParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("l2"))
			this.l2 = Double.valueOf(parameterValue);
		else if (parameter.equals("epsilon"))
			this.epsilon = Double.valueOf(parameterValue);
		else
			return false;
		return true;
	}
	
	public SupervisedModel<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		SupervisedModelSVMDeadPegasos<D, L> clone = (SupervisedModelSVMDeadPegasos<D, L>)super.clone(datumTools, environment);
		
		clone.labelIndices = this.labelIndices;
		clone.trainingIterations = this.trainingIterations;
		
		return clone;
	}
	
	@Override
	protected boolean deserializeParameters(BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		Pair<String, String> tAssign = SerializationUtil.deserializeAssignment(reader);
		Pair<String, String> sAssign = SerializationUtil.deserializeAssignment(reader);
		Pair<String, String> numWeightsAssign = SerializationUtil.deserializeAssignment(reader);
	
		int numWeights = Integer.valueOf(numWeightsAssign.getSecond());
		this.numFeatures = numWeights / this.labelIndices.size();
		
		this.t = Integer.valueOf(tAssign.getSecond());
		this.s = Double.valueOf(sAssign.getSecond());
		this.featureNames = new HashMap<Integer, String>();
		
		this.feature_W = new double[numWeights];
		
		this.bias_b = new double[this.labelIndices.size()];
			
		String assignmentLeft = null;
		while ((assignmentLeft = SerializationUtil.deserializeAssignmentLeft(reader)) != null) {
			if (assignmentLeft.equals("labelFeature")) {
				String labelFeature = SerializationUtil.deserializeGenericName(reader);
				Map<String, String> featureParameters = SerializationUtil.deserializeArguments(reader);
				
				String featureName = labelFeature.substring(labelFeature.indexOf("-") + 1);
				double W = Double.valueOf(featureParameters.get("W"));
				int labelIndex = Integer.valueOf(featureParameters.get("labelIndex"));
				int featureIndex = Integer.valueOf(featureParameters.get("featureIndex"));
				
				int index = labelIndex*this.numFeatures+featureIndex;
				this.featureNames.put(featureIndex, featureName);
				this.feature_W[index] = W;
			} else if (assignmentLeft.equals("labelBias")) {
				SerializationUtil.deserializeGenericName(reader);
				Map<String, String> biasParameters = SerializationUtil.deserializeArguments(reader);
				double b = Double.valueOf(biasParameters.get("b"));
				int index = Integer.valueOf(biasParameters.get("index"));
				
				this.bias_b[index] = b;
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

		writer.write("\n");
		
		return true;
	}

	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelSVMDeadPegasos<D, L>();
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
	
	@Override
	public Map<D, L> classify(FeaturizedDataSet<D, L> data) {
		Map<D, L> classifiedData = new HashMap<D, L>();
		
		for (D datum : data) {
			classifiedData.put(datum, argMaxScoreLabel(data, datum, false));
		}
	
		return classifiedData;
	}
}
