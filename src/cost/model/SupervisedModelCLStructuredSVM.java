package cost.model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.annotation.structure.DatumStructure;
import ark.data.annotation.structure.DatumStructureCollection;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.util.Pair;
import ark.util.SerializationUtil;

/**
 * SupervisedModelCLStructuredSVM is an implementation of a structured 
 * cost learning SVM that minimizes a structured version of 
 * objective function (1) in the paper/previous-approaches.pdf 
 * document. This implementation is deprecated as it extends from
 * the deprecated cost.model.SupervisedModelCL. 
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> label type
 * 
 * @deprecated Use ark.model.SupervisedModelSVMStructured as a
 * structured SVM instead.  There is currently no recently implemented
 * structured cost learning SVM, and so if you want one, then you should
 * implement it by extending either ark.model.SupervisedModelSVM or
 * ark.model.SupervisedModelSVMStructured, depending on how you design it.
 *
 */
public class SupervisedModelCLStructuredSVM<D extends Datum<L>, L> extends SupervisedModelCL<D, L> {	
	private CostWeightComparator costWeightComparator;
	protected double[] feature_g;
	protected double[] bias_g;
	protected double[] cost_g;
	protected int iteration;
	
	protected String datumStructureOptimizer;
	protected String datumStructureCollection;
	protected DatumStructureCollection<D, L> trainingDatumStructureCollection;
	
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
	
	public SupervisedModelCLStructuredSVM() {
		super();
		this.featureNames = new HashMap<Integer, String>();
	}

	@Override
	protected boolean initializeTraining(FeaturizedDataSet<D, L> data) {		
		this.costWeightComparator = new CostWeightComparator();
		this.feature_g = new double[this.feature_w.length];
		this.bias_g = new double[this.bias_b.length];
		this.cost_g = new double[this.cost_v.length];
		
		this.trainingDatumStructureCollection = data.getDatumTools().makeDatumStructureCollection(this.datumStructureCollection, data);
		
		return true;
	}
	
	@Override
	protected boolean trainOneIteration(FeaturizedDataSet<D, L> data) {
		for (DatumStructure<D, L> datumStructure : this.trainingDatumStructureCollection) {
			Map<D, Map<L, Double>> scoredDatumLabels = scoreDatumStructureLabels(data, datumStructure, true);
			Map<D, L> datumLabels = datumStructure.getDatumLabels(this.labelMapping);
			// Maybe just optimize here...?
			Map<D, L> bestDatumLabels = getBestDatumLabels(data, datumStructure, scoredDatumLabels);

			Map<Integer, Double> datumStructureFeatureValues = computeDatumStructureFeatureValues(data, datumStructure, datumLabels, iteration == 0);
			Map<Integer, Double> bestStructureFeatureValues = computeDatumStructureFeatureValues(data, datumStructure, bestDatumLabels, false);
			Map<Integer, Double> bestStructureCosts = computeDatumStructureCosts(datumStructure, bestDatumLabels);
			
			// Update feature weights
			for (int i = 0; i < this.feature_w.length; i++) { 
				double datumFeatureValue = (datumStructureFeatureValues.containsKey(i)) ? datumStructureFeatureValues.get(i) : 0.0;
				double bestFeatureValue = (bestStructureFeatureValues.containsKey(i)) ? bestStructureFeatureValues.get(i) : 0.0;
				
				if (this.l1 == 0 && this.feature_w[i] == 0 && datumFeatureValue == bestFeatureValue)
					continue;
				
				feature_g[i] = this.l2*this.feature_w[i]-datumFeatureValue+bestFeatureValue;
				
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
			
			// Update label biases
			for (int i = 0; i < this.bias_b.length; i++) {
				L label = this.labelIndices.reverseGet(i);
				int datumLabelCount = getLabelCount(datumLabels, label);
				int bestLabelCount = getLabelCount(bestDatumLabels, label);
				bias_g[i] = -datumLabelCount + bestLabelCount;
				
				this.bias_G[i] += bias_g[i]*bias_g[i];
				this.bias_u[i] += bias_g[i];
				if (this.bias_G[i] == 0)
					continue;
				this.bias_b[i] -= bias_g[i]*this.n/Math.sqrt(this.bias_G[i]);
			}
			
			// Update cost weights
			for (int i = 0; i < this.cost_v.length; i++) {
				cost_g[i] = (bestStructureCosts.containsKey(i)) ? bestStructureCosts.get(i) : 0;
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
	
	private int getLabelCount(Map<D, L> datumsToLabels, L countLabel) {
		int count = 0;
		for (L label : datumsToLabels.values())
			if (label.equals(countLabel))
				count++;
		return count;
	}
	
	private Map<D, Map<L, Double>> scoreDatumStructureLabels(FeaturizedDataSet<D, L> data, DatumStructure<D, L> datumStructure, boolean includeCost) {
		Map<D, Map<L, Double>> datumLabelScores = new HashMap<D, Map<L, Double>>();
		
		for (D datum : datumStructure) {
			Map<L, Double> scores = new HashMap<L, Double>();
			
			double max = Double.NEGATIVE_INFINITY;
			for (L label : this.validLabels) {
				double score = scoreDatumLabel(data, datum, label, includeCost);
				scores.put(label, score);
				if (score > max)
					max = score;
			}
			
			// Map to simplex
			/*double lse = 0;
			for (Double score : scores.values())
				lse += Math.exp(score - max);
			lse = max + Math.log(lse);
		
			for (Entry<L, Double> scoreEntry : scores.entrySet()) {
				scoreEntry.setValue(Math.exp(scoreEntry.getValue()-lse));
			}*/
			
			datumLabelScores.put(datum, scores);
		}
		
		return datumLabelScores;
	}
	
	private Map<Integer, Double> computeDatumStructureCosts(DatumStructure<D, L> datumStructure, Map<D, L> labels) {
		Map<Integer, Double> costs = new HashMap<Integer, Double>();
		
		for (D datum : datumStructure) {
			Map<Integer, Double> datumCosts = this.factoredCost.computeVector(datum, labels.get(datum));
			for (Entry<Integer, Double> entry : datumCosts.entrySet()) {
				if (!costs.containsKey(entry.getKey()))
					costs.put(entry.getKey(), 0.0);
				costs.put(entry.getKey(), costs.get(entry.getKey()) + entry.getValue());
			}
		}
		
		return costs;
	}
	
	private double scoreDatumStructure(FeaturizedDataSet<D, L> data, DatumStructure<D, L> datumStructure, Map<D, L> structureLabels, boolean includeCost) {
		double score = 0.0;
		
		Map<Integer, Double> datumStructureFeatureValues = computeDatumStructureFeatureValues(data, datumStructure, structureLabels, false);
		for (Entry<Integer, Double> entry : datumStructureFeatureValues.entrySet()) {
			score += this.feature_w[entry.getKey()]*entry.getValue();
		}
		
		for (int i = 0; i < this.bias_b.length; i++) {
			L label = this.labelIndices.reverseGet(i);
			int datumLabelCount = getLabelCount(structureLabels, label);
			score += this.bias_b[i]*datumLabelCount;
		}
		
		if (includeCost) {
			Map<Integer, Double> datumStructureCosts = computeDatumStructureCosts(datumStructure, structureLabels);
			for (Entry<Integer, Double> entry : datumStructureCosts.entrySet()) {
				score += this.cost_v[entry.getKey()]*entry.getValue();
			}
		}
		
		return score;
	}
	
	private double scoreDatumLabel(FeaturizedDataSet<D, L> data, D datum, L label, boolean includeCost) {
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

	private Map<Integer, Double> computeDatumStructureFeatureValues(FeaturizedDataSet<D,L> data, DatumStructure<D, L> datumStructure, Map<D, L> structureLabels, boolean cacheFeatureNames) {
		Map<Integer, Double> featureValues = new HashMap<Integer, Double>();
		int numDatumFeatures = data.getFeatureVocabularySize();
		for (D datum : datumStructure) {
			Map<Integer, Double> datumFeatureValues = data.getFeatureVocabularyValues(datum);
			int labelIndex = this.labelIndices.get(structureLabels.get(datum));
			int featureLabelOffset = numDatumFeatures*labelIndex;
			
			for (Entry<Integer, Double> entry : datumFeatureValues.entrySet()) {
				int featureIndex = featureLabelOffset + entry.getKey();
				if (!featureValues.containsKey(featureIndex))
					featureValues.put(featureIndex, 0.0);
				featureValues.put(featureIndex, featureValues.get(featureIndex) + entry.getValue());
			}
			
			if (cacheFeatureNames) {
				List<Integer> missingNameKeys = new ArrayList<Integer>();
				for (Integer key : datumFeatureValues.keySet())
					if (!this.featureNames.containsKey(key))
						missingNameKeys.add(key);
				this.featureNames.putAll(data.getFeatureVocabularyNamesForIndices(missingNameKeys));				
			}
		}
		
		return featureValues;
	}

	private Map<D, L> getBestDatumLabels(FeaturizedDataSet<D, L> data, DatumStructure<D, L> datumStructure, Map<D, Map<L, Double>> scoredDatumLabels) {
		Map<D, L> optimizedDatumLabels = datumStructure.optimize(this.datumStructureOptimizer, scoredDatumLabels, this.fixedDatumLabels, this.validLabels, this.labelMapping);
		Map<D, L> actualDatumLabels = datumStructure.getDatumLabels(this.labelMapping);
		
		double optimizedScore = scoreDatumStructure(data, datumStructure, optimizedDatumLabels, true);
		double actualScore = scoreDatumStructure(data, datumStructure, actualDatumLabels, false);
		
		if (actualScore > optimizedScore)
			return actualDatumLabels;
		else
			return optimizedDatumLabels;
	}
	
	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> posteriors = new HashMap<D, Map<L, Double>>(data.size());
		DatumStructureCollection<D, L> datumStructureCollection = data.getDatumTools().makeDatumStructureCollection(this.datumStructureCollection, data);
		Map<D, L> bestDatumLabels = new HashMap<D, L>();
		
		if (this.factoredCost != null && !this.factoredCost.init(this, data))
			return null;

		
		for (DatumStructure<D, L> datumStructure : datumStructureCollection) {
			Map<D, Map<L, Double>> scoredDatumLabels = scoreDatumStructureLabels(data, datumStructure, false);
			bestDatumLabels.putAll(
				datumStructure.optimize(this.datumStructureOptimizer, scoredDatumLabels, this.fixedDatumLabels, this.validLabels, this.labelMapping)
			);
		}
		
		for (D datum : data) {
			posteriors.put(datum, new HashMap<L, Double>());
			L bestLabel = bestDatumLabels.get(datum);
			
			if (bestLabel == null) {
				double p = 1.0/this.validLabels.size();
				posteriors.get(datum).put(bestLabel, p);
			} else {
				for (L label : this.validLabels) {
					if (label.equals(bestLabel))
						posteriors.get(datum).put(label, 1.0);
					else
						posteriors.get(datum).put(label, 0.0);
				}
			}
		}
		
		return posteriors;
	}
	
	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelCLStructuredSVM<D, L>();
	}
	
	@Override
	public String getGenericName() {
		return "CLStructuredSVM";
	}

	@Override
	public double computeLoss(FeaturizedDataSet<D, L> data) {
		double loss = 0;
		
		DatumStructureCollection<D, L> datumStructureCollection = data.getDatumTools().makeDatumStructureCollection(this.datumStructureCollection, data);
		
		for (DatumStructure<D, L> datumStructure : datumStructureCollection) {
			Map<D, Map<L, Double>> scoredDatumLabels = scoreDatumStructureLabels(data, datumStructure, true);
			Map<D, L> datumLabels = datumStructure.getDatumLabels(this.labelMapping);
			Map<D, L> bestDatumLabels = getBestDatumLabels(data, datumStructure, scoredDatumLabels);
		
			double datumStructureScore = scoreDatumStructure(data, datumStructure, datumLabels, false);
			double bestStructureScore = scoreDatumStructure(data, datumStructure, bestDatumLabels, true);
		
			loss += bestStructureScore - datumStructureScore;
		}
		
		return loss;
	}
	
	public SupervisedModel<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		SupervisedModelCLStructuredSVM<D, L> clone = (SupervisedModelCLStructuredSVM<D, L>)super.clone(datumTools, environment);
		
		clone.labelIndices = this.labelIndices;
		clone.trainingIterations = this.trainingIterations;
		clone.datumStructureCollection = this.datumStructureCollection;
		clone.datumStructureOptimizer = this.datumStructureOptimizer;
		if (this.factoredCost != null) {
			clone.factoredCost = this.factoredCost.clone(datumTools, environment);
		}
		
		return clone;
	}
	
	@Override
	protected boolean deserializeExtraInfo(String name, BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		if (name.equals("datumStructureCollection")) {
			this.datumStructureCollection = SerializationUtil.deserializeAssignmentRight(reader);
		} else if (name.equals("datumStructureOptimizer")) {
			this.datumStructureOptimizer = SerializationUtil.deserializeAssignmentRight(reader);
		} else {
			return super.deserializeExtraInfo(name, reader, datumTools);
		}
		
		return true;
	}

	@Override
	protected boolean serializeExtraInfo(Writer writer) throws IOException {		
		if (this.datumStructureCollection != null) {
			writer.write("\t");
			Pair<String, String> datumStructureCollectionAssignment = new Pair<String, String>("datumStructureCollection", this.datumStructureCollection);
			if (!SerializationUtil.serializeAssignment(datumStructureCollectionAssignment, writer))
				return false;
			writer.write("\n");
		}
		
		if (this.datumStructureOptimizer != null) {
			writer.write("\t");
			Pair<String, String> datumStructureOptimizerAssignment = new Pair<String, String>("datumStructureOptimizer", this.datumStructureOptimizer);
			if (!SerializationUtil.serializeAssignment(datumStructureOptimizerAssignment, writer))
				return false;
			writer.write("\n");
		}
		
		return super.serializeExtraInfo(writer);
	}
}

