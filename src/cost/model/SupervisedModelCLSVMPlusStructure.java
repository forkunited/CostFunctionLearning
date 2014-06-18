package cost.model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
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

public class SupervisedModelCLSVMPlusStructure<D extends Datum<L>, L> extends SupervisedModelCLSVM<D, L> {	
	protected String datumStructureOptimizer;
	protected String datumStructureCollection;
	
	public SupervisedModelCLSVMPlusStructure() {
		super();
		this.featureNames = new HashMap<Integer, String>();
	}
	
	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> datumPosteriors = new HashMap<D, Map<L, Double>>(data.size());
		if (this.factoredCost != null && !this.factoredCost.init(this, data))
			return null;
		for (D datum : data) {
			datumPosteriors.put(datum, posteriorForDatum(data, datum));
		}
		
		DatumStructureCollection<D, L> datumStructureCollection = data.getDatumTools().makeDatumStructureCollection(this.datumStructureCollection, data);
		Map<D, Map<L, Double>> structurePosteriors = new HashMap<D, Map<L, Double>>(data.size());
		
		for (DatumStructure<D, L> datumStructure : datumStructureCollection) {
			Map<D, L> optimizedDatumLabels = datumStructure.optimize(this.datumStructureOptimizer, datumPosteriors, this.fixedDatumLabels, this.validLabels, this.labelMapping);
			for (Entry<D, L> entry : optimizedDatumLabels.entrySet()) {
				Map<L, Double> p = new HashMap<L, Double>();
				for (L validLabel : this.validLabels) {
					p.put(validLabel, 0.0);
				}
				p.put(entry.getValue(), 1.0);
				
				structurePosteriors.put(entry.getKey(), p);
			}
		}

		return structurePosteriors;
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
		return new SupervisedModelCLSVMPlusStructure<D, L>();
	}
	
	@Override
	public String getGenericName() {
		return "CLSVMPlusStructure";
	}

	public SupervisedModel<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		SupervisedModelCLSVMPlusStructure<D, L> clone = (SupervisedModelCLSVMPlusStructure<D, L>)super.clone(datumTools, environment);
		
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

